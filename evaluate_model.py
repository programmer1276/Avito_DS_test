"""Повторная оценка зафиксированной модели на отложенных 1500 запросах.

Пример:
  python evaluate_model.py --data-dir data --model-dir models \
      --cache-dir cache/train --output-dir reports/holdout --device cpu

Скрипт восстанавливает заранее заданное разбиение, сравнивает выбранную модель
с baseline и сохраняет макро Recall@50 и парный bootstrap. Подбора параметров
здесь нет. Импорт модуля и --help не читают данные и не запускают оценку.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import pickle
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from train_model import digest_texts, resolve, write_json


def bootstrap_metrics(baseline, selected, repetitions=3000, seed=20260929):
    """Один и тот же набор запросов в обеих моделях каждой репликации."""
    baseline = np.asarray(baseline, dtype=np.float64)
    selected = np.asarray(selected, dtype=np.float64)
    if baseline.shape != selected.shape or baseline.ndim != 1 or not len(baseline):
        raise ValueError('Нужны два непустых вектора Recall одинакового размера')
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(selected), size=(repetitions, len(selected)))
    model_sample = selected[indices].mean(axis=1)
    difference_sample = (selected-baseline)[indices].mean(axis=1)
    return {
        'model_bootstrap_95_ci': np.quantile(model_sample, [.025, .975]).tolist(),
        'paired_improvement_bootstrap_95_ci': np.quantile(difference_sample, [.025, .975]).tolist(),
        'bootstrap_repetitions': repetitions, 'bootstrap_seed': seed,
    }


def evaluate_model(data_dir, model_dir, cache_dir, output_dir, device='cpu'):
    # Все действия с данными находятся внутри явного вызова evaluate_model.
    from retrieval import SEARCH_COLS, normalize
    from splits import make_v2_splits
    from solution_pipeline import (
        ENCODER_REVISION, LocalE5Encoder, item_passages, clean, build_engine,
        batched_cosines, candidate_pool, candidate_cosines, buildfeatures,
        feature_names, score_candidates, top_indices,
    )

    started = time.time()
    data_dir = Path(data_dir).expanduser().resolve()
    model_dir = Path(model_dir).expanduser().resolve()
    cache_dir = Path(cache_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = model_dir/'selected_config.json'
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    views = config.get('views', ['title', 'full'])
    if views not in [['title'], ['title', 'full']]:
        raise ValueError('Порядок признаков должен быть [title] или [title, full]')
    if config.get('query_mode', 'plain64') != 'plain64':
        raise ValueError('Нужен исходный текст запроса с ограничением 64 токена')
    pool_type = config.get('pool', 'expanded')
    if pool_type not in ['old', 'expanded', 'fullpool']:
        raise ValueError('Неизвестный пул кандидатов')
    revision = config.get('encoder_revision', ENCODER_REVISION)
    intent_config = config.get('intent', {'alpha': .2, 'use_filters': True})
    ranker_path = resolve(model_dir, config.get('ranker_file', 'ranker.pkl'))
    ranker_bytes = ranker_path.read_bytes()
    ranker_hash = hashlib.sha256(ranker_bytes).hexdigest()
    model = pickle.loads(ranker_bytes)
    del ranker_bytes
    names_path = resolve(model_dir, config.get('feature_names_file', 'feature_names.json'))
    expected_names = json.loads(names_path.read_text(encoding='utf-8'))

    # Корпус содержит признаки всех объявлений из train. Разметка holdout
    # исключается из статистик и обучения, но сами документы доступны поиску.
    opened_at = datetime.now(timezone.utc).isoformat()
    train_path = data_dir/'train.parquet'
    train = pd.read_parquet(train_path)
    item_columns = [name for name in train.columns if name.startswith('item_')]
    items = train[item_columns].drop_duplicates('item_id').reset_index(drop=True)
    reference, rank_queries, dev_queries, holdout = make_v2_splits(train)
    del train
    if len(holdout) != 1500:
        raise ValueError('Ожидалось заранее отложенное множество из 1500 запросов')
    held_texts = set(holdout.search_query.map(normalize))
    if len(held_texts) != len(holdout):
        raise ValueError('Нормализованные тексты holdout должны быть уникальны')
    for name, frame in [('reference', reference), ('rank_train', rank_queries), ('dev', dev_queries)]:
        if held_texts & set(frame.search_query.map(normalize)):
            raise ValueError(f'Тексты holdout пересекаются с {name}')
    split_counts = {'reference_rows': len(reference), 'rank_train_queries': len(rank_queries),
                    'dev_queries': len(dev_queries), 'holdout_queries': len(holdout)}
    print('Разбиение:', json.dumps(split_counts, ensure_ascii=False), flush=True)
    reference = reference[SEARCH_COLS+['item_id', 'item_location_id', 'item_microcat_id']]
    del rank_queries, dev_queries

    # Формат этих метаданных совпадает с train_model.py: при одинаковом коде,
    # данных и порядке ID можно переиспользовать его индекс и эмбеддинги.
    script_dir = Path(__file__).resolve().parent
    code_files = ['retrieval.py', 'extra_retrieval.py', 'intent_model.py',
                  'rerank_model.py', 'splits.py', 'baseline_utils.py', 'solution_pipeline.py']
    code_hashes = {name: hashlib.sha256((script_dir/name).read_bytes()).hexdigest()
                   for name in code_files}
    item_ids_hash = digest_texts(items.item_id)
    engine_signature = {
        'training_source': {'path': str(train_path), 'size': train_path.stat().st_size,
                            'mtime_ns': train_path.stat().st_mtime_ns},
        'code_hashes': code_hashes, 'intent': intent_config, 'split_seed': 20260929,
        'item_ids_sha256': item_ids_hash,
    }
    engine_path, engine_meta = cache_dir/'train_engine.pkl', cache_dir/'train_engine.json'
    if (engine_path.exists() and engine_meta.exists()
            and json.loads(engine_meta.read_text()) == engine_signature):
        with engine_path.open('rb') as file:
            engine = pickle.load(file)
    else:
        engine = build_engine(items, reference, intent_config)
        with engine_path.open('wb') as file:
            pickle.dump(engine, file, protocol=5)
        write_json(engine_meta, engine_signature)
    if not np.array_equal(engine.base.ids, items.item_id.to_numpy()):
        raise ValueError('Порядок объявлений в кэше отличается от корпуса')
    del reference

    required_views = list(views)
    if pool_type == 'fullpool' and 'full' not in required_views:
        required_views.append('full')
    encoder = None
    encoder_dir = resolve(model_dir, config.get('encoder_dir', 'e5-small'))
    vectors = {}
    for view in required_views:
        vector_path, meta_path = cache_dir/f'items_{view}.npy', cache_dir/f'items_{view}.json'
        metadata = {'text_sha256': digest_texts(item_passages(items, view)),
                    'revision': revision, 'view': view,
                    'max_length': 64 if view == 'title' else 128,
                    'item_ids_sha256': item_ids_hash}
        if not (vector_path.exists() and meta_path.exists()
                and json.loads(meta_path.read_text()) == metadata):
            if encoder is None:
                encoder = LocalE5Encoder(encoder_dir, device=device, revision=revision)
            print(f'Кодирование объявлений: {view}', flush=True)
            np.save(vector_path, encoder.encode_items(items, view))
            write_json(meta_path, metadata)
        vectors[view] = np.load(vector_path, mmap_mode='r')
        if vectors[view].shape != (len(items), 384):
            raise ValueError('Неверный размер матрицы эмбеддингов объявлений')

    query_path, query_meta = cache_dir/'holdout_queries.npy', cache_dir/'holdout_queries.json'
    metadata = {'query_sha256': digest_texts(holdout.search_query.map(clean)),
                'revision': revision, 'max_length': 64, 'query_mode': 'plain64'}
    if not (query_path.exists() and query_meta.exists()
            and json.loads(query_meta.read_text()) == metadata):
        if encoder is None:
            encoder = LocalE5Encoder(encoder_dir, device=device, revision=revision)
        np.save(query_path, encoder.encode_queries(holdout))
        write_json(query_meta, metadata)
    query_vectors = np.load(query_path, mmap_mode='r')
    if query_vectors.shape != (len(holdout), 384):
        raise ValueError('Неверный размер матрицы эмбеддингов запросов')
    del encoder

    rows = []
    records = holdout.to_dict('records')
    for begin, title_batch in batched_cosines(query_vectors, vectors['title'], batch_size=32):
        full_batch = None
        if pool_type == 'fullpool':
            with threadpool_limits(limits=3):
                full_batch = np.asarray(query_vectors[begin:begin+len(title_batch)] @ vectors['full'].T,
                                        dtype=np.float32)
        for offset, title_retrieval_cos in enumerate(title_batch):
            number = begin+offset
            q = records[number]
            truth_ids = set(q['item_id'])
            if not truth_ids or not truth_ids <= engine.base.id_to_idx.keys():
                raise ValueError('Все положительные объявления должны быть в корпусе')
            truth = {engine.base.id_to_idx[item] for item in truth_ids}
            idx, parts, extra, sources, old = candidate_pool(
                engine, q, title_retrieval_cos, pool=pool_type,
                fullcos=None if full_batch is None else full_batch[offset])
            # Ретривер считает cosine блочным BLAS; признаки повторяют einsum
            # обучения, чтобы не менять значения на границах разбиений деревьев.
            title_cos = candidate_cosines(vectors['title'], query_vectors[number], idx)
            full_cos = (candidate_cosines(vectors['full'], query_vectors[number], idx)
                        if 'full' in views else None)
            X = buildfeatures(engine, q, idx, parts, extra, sources, title_cos, full_cos)
            names = feature_names(engine, views)
            if names != expected_names:
                raise ValueError('Схема признаков не совпадает со схемой модели')
            score = score_candidates(model, X, names,
                                     baseline_log_weight=config.get('baseline_log_weight', 0.),
                                     sampling_correction=config.get('sampling_correction', 0.),
                                     old_pool_size=len(old))
            selected = idx[top_indices(score, 50)]
            baseline = top_indices(sources['baseline'], 50)
            if len(selected) != 50 or len(set(selected)) != 50:
                raise ValueError('Нужно ровно 50 различных кандидатов')
            selected_hits = len(set(selected) & truth)
            baseline_hits = len(set(baseline) & truth)
            pool_hits = len(set(idx) & truth)
            row = {key: value for key, value in q.items() if key != 'item_id'}
            row.update(query_number=number, n_positives=len(truth), pool_size=len(idx),
                       baseline_hits=baseline_hits, model_hits=selected_hits, pool_hits=pool_hits,
                       baseline_recall=baseline_hits/len(truth), model_recall=selected_hits/len(truth),
                       pool_recall=pool_hits/len(truth), positives=' '.join(sorted(truth_ids)),
                       baseline_top50=' '.join(engine.base.ids[baseline]),
                       model_top50=' '.join(engine.base.ids[selected]))
            rows.append(row)
            if (number+1) % 150 == 0:
                print(f'Оценка: {number+1}/{len(records)}, {time.time()-started:.1f} с', flush=True)

    frame = pd.DataFrame(rows)
    baseline, selected = frame.baseline_recall.to_numpy(), frame.model_recall.to_numpy()
    difference = selected-baseline
    # Артефакты могут быть открыты другим процессом: обнаруживаем замену
    # конфигурации или модели во время расчёта и не сохраняем смешанную оценку.
    if config_path.read_bytes() != config_bytes or hashlib.sha256(ranker_path.read_bytes()).hexdigest() != ranker_hash:
        raise RuntimeError('Модель или конфигурация изменились во время оценки')
    report = {
        'protocol': 'Evaluation of a fixed configuration on the predefined 1500 unseen query texts; no parameter selection.',
        'config': config, 'config_sha256': hashlib.sha256(config_bytes).hexdigest(),
        'ranker_sha256': ranker_hash, 'data_opened_at_utc': opened_at, 'device': device,
        'split_seed': 20260929, **split_counts, 'queries': len(frame),
        'positives': int(frame.n_positives.sum()), 'corpus_items': len(items),
        'baseline_macro_recall50': float(baseline.mean()),
        'model_macro_recall50': float(selected.mean()),
        'pool_macro_recall': float(frame.pool_recall.mean()),
        'macro_improvement': float(difference.mean()),
        **bootstrap_metrics(baseline, selected),
        'improved_queries': int(np.sum(difference > 0)),
        'worsened_queries': int(np.sum(difference < 0)), 'equal_queries': int(np.sum(difference == 0)),
        'model_full_recall_queries': int(np.sum(selected == 1)),
        'model_partial_recall_queries': int(np.sum((selected > 0) & (selected < 1))),
        'model_zero_recall_queries': int(np.sum(selected == 0)),
        'mean_pool_size': float(frame.pool_size.mean()), 'seconds': time.time()-started,
        'limitation': 'Local corpus and query distribution differ from benchmark; this interval does not predict the hidden benchmark score.',
    }
    frame.to_csv(output_dir/'fresh_holdout_per_query.csv', index=False, encoding='utf-8')
    write_json(output_dir/'fresh_holdout_metrics.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', required=True, help='Каталог с исходным train.parquet')
    parser.add_argument('--model-dir', required=True, help='Каталог selected_config.json, ranker и локальных весов E5')
    parser.add_argument('--cache-dir', required=True, help='Можно указать кэш train_model.py для повторного использования')
    parser.add_argument('--output-dir', required=True, help='Каталог метрик и результатов по запросам')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda'], help='Для повторения сохранённых результатов: cpu')
    args = parser.parse_args()
    evaluate_model(args.data_dir, args.model_dir, args.cache_dir, args.output_dir, args.device)


if __name__ == '__main__':
    main()
