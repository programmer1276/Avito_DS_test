"""Воспроизведение обучения выбранной модели из исходного train.parquet.

Пример:
  python train_model.py --data-dir data --model-dir models --cache-dir cache/train

Параметры берутся из models/selected_config.json. Скрипт не подбирает их заново
и не оценивает holdout. E5 загружается только из локального каталога модели.
"""
import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def digest_texts(texts):
    """Хеш учитывает содержимое и границы строк, поэтому порядок важен."""
    digest = hashlib.sha256()
    for text in texts:
        encoded = str(text).encode('utf-8')
        digest.update(len(encoded).to_bytes(8, 'little'))
        digest.update(encoded)
    return digest.hexdigest()


def resolve(root, filename):
    path = Path(filename).expanduser()
    return path if path.is_absolute() else root/path


def train_model(data_dir, model_dir, cache_dir, device='cpu'):
    # Отложенные импорты позволяют вызвать --help без загрузки нейросети.
    from retrieval import normalize
    from splits import make_v2_splits
    from solution_pipeline import (
        ENCODER_REVISION, LocalE5Encoder, item_passages, build_engine,
        batched_cosines, candidate_pool, candidate_cosines, top_indices,
        feature_names,
    )

    started = time.time()
    data_dir = Path(data_dir).expanduser().resolve()
    model_dir = Path(model_dir).expanduser().resolve()
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((model_dir/'selected_config.json').read_text(encoding='utf-8'))
    views = config.get('views', ['title', 'full'])
    if views not in [['title'], ['title', 'full']]:
        raise ValueError('Ожидаются views=[title] или [title, full] в указанном порядке')
    if config.get('training_pool', 'expanded') != 'expanded':
        raise ValueError('Этот скрипт воспроизводит обучение на expanded pool')
    if config.get('pool', 'expanded') not in ['expanded', 'fullpool']:
        raise ValueError('Неизвестный пул предсказания; обучение всегда использует expanded')
    if config.get('query_mode', 'plain64') != 'plain64':
        raise ValueError('Нужен исходный текст запроса с ограничением 64 токена')
    revision = config.get('encoder_revision', ENCODER_REVISION)
    intent_config = config.get('intent', {'alpha': .2, 'use_filters': True})
    hyperparameters = dict(config['hyperparameters'])
    expected_parameters = {'max_leaf_nodes', 'learning_rate', 'l2_regularization',
                           'max_iter', 'min_samples_leaf', 'max_bins', 'random_state'}
    if set(hyperparameters) != expected_parameters:
        raise ValueError(f'hyperparameters должны содержать {sorted(expected_parameters)}')

    train_path = data_dir/'train.parquet'
    train = pd.read_parquet(train_path)
    item_columns = [name for name in train.columns if name.startswith('item_')]
    items = train[item_columns].drop_duplicates('item_id').reset_index(drop=True)
    reference, rank_queries, _, _ = make_v2_splits(train)
    # В split-функции holdout только отделяется. Его ответы здесь не читаются,
    # не оцениваются и не используются ни для статистик, ни для бустинга.
    split_report = {'reference_rows': len(reference),
                    'reference_texts': int(reference.search_query.map(normalize).nunique()),
                    'rank_train_queries': len(rank_queries),
                    'item_count': len(items), 'split_seed': 20260929}
    print('Разбиение:', json.dumps(split_report, ensure_ascii=False), flush=True)

    code_files = ['retrieval.py', 'extra_retrieval.py', 'intent_model.py',
                  'rerank_model.py', 'splits.py', 'baseline_utils.py', 'solution_pipeline.py']
    script_dir = Path(__file__).resolve().parent
    code_hashes = {name: hashlib.sha256((script_dir/name).read_bytes()).hexdigest()
                   for name in code_files}
    signature = {'training_source': {'path': str(train_path), 'size': train_path.stat().st_size,
                                     'mtime_ns': train_path.stat().st_mtime_ns},
                 'code_hashes': code_hashes, 'intent': intent_config,
                 'revision': revision, 'views': views, 'pool': 'expanded',
                 'split_seed': 20260929, 'sampling_seed': 20260930,
                 'hard_count': 70, 'random_count': 120,
                 'positive_policy': 'all known positives already in retrieved pool',
                 'item_ids_sha256': digest_texts(items.item_id),
                 'rank_query_order_sha256': digest_texts(rank_queries.search_query)}

    feature_path = cache_dir/'training_features.npz'
    feature_meta = cache_dir/'training_features.json'
    use_features = (feature_path.exists() and feature_meta.exists()
                    and json.loads(feature_meta.read_text())['signature'] == signature)
    if use_features:
        print('Читаются ранее рассчитанные признаки с совпадающей сигнатурой.', flush=True)
        saved = np.load(feature_path, allow_pickle=False)
        X, y, groups, inclusion = [saved[k] for k in ['X', 'y', 'group', 'inclusion']]
        pool_recall = saved['pool_recall']
        names = json.loads(feature_meta.read_text())['feature_names']
    else:
        engine_path, engine_meta = cache_dir/'train_engine.pkl', cache_dir/'train_engine.json'
        engine_signature = {key: signature[key] for key in
                            ['training_source', 'code_hashes', 'intent', 'split_seed', 'item_ids_sha256']}
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
            raise ValueError('Порядок объявлений в кэше индекса не совпадает с корпусом')

        encoder = None
        vectors = {}
        encoder_dir = resolve(model_dir, config.get('encoder_dir', 'e5-small'))
        for view in views:
            vector_path, meta_path = cache_dir/f'items_{view}.npy', cache_dir/f'items_{view}.json'
            text_hash = digest_texts(item_passages(items, view))
            metadata = {'text_sha256': text_hash, 'revision': revision, 'view': view,
                        'max_length': 64 if view == 'title' else 128,
                        'item_ids_sha256': signature['item_ids_sha256']}
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

        query_path, query_meta = cache_dir/'rank_queries.npy', cache_dir/'rank_queries.json'
        from solution_pipeline import clean
        metadata = {'query_sha256': digest_texts(rank_queries.search_query.map(clean)),
                    'revision': revision, 'max_length': 64, 'query_mode': 'plain64'}
        if not (query_path.exists() and query_meta.exists()
                and json.loads(query_meta.read_text()) == metadata):
            if encoder is None:
                encoder = LocalE5Encoder(encoder_dir, device=device, revision=revision)
            np.save(query_path, encoder.encode_queries(rank_queries))
            write_json(query_meta, metadata)
        query_vectors = np.load(query_path, mmap_mode='r')
        if query_vectors.shape != (len(rank_queries), 384):
            raise ValueError('Неверный размер матрицы эмбеддингов запросов')
        del encoder

        records = rank_queries.to_dict('records')
        rng = np.random.default_rng(20260930)
        xs, ys, group_parts, probabilities = [], [], [], []
        recalls = []
        # Матрица сходства считается небольшими блоками, а не целиком.
        for first, title_scores in batched_cosines(query_vectors, vectors['title'],
                                                   batch_size=32, blas_threads=4):
            for offset, titlecos in enumerate(title_scores):
                j = first+offset
                query = records[j]
                pool, parts, extra, sources, _ = candidate_pool(engine, query, titlecos, pool='expanded')
                truth = {engine.base.id_to_idx[item_id] for item_id in query['item_id']}
                retrieved_truth = pool[np.isin(pool, list(truth))]
                hard = pool[top_indices(sources['baseline'][pool], 70)]
                rest = np.setdiff1d(pool, hard)
                random_part = rng.choice(rest, min(120, len(rest)), replace=False)
                selected = np.unique(np.concatenate([hard, random_part, retrieved_truth]))
                positions = np.searchsorted(pool, selected)
                # Ранги и относительные оценки определены по ПОЛНОМУ пулу.
                # Считать engine.build только на выбранных 190 строках нельзя.
                features = engine.build(query, pool, parts, extra, sources)[positions]
                geo = parts['geo'][selected]
                category = np.where(parts['category'][selected] > 0, 1., .3)
                for view in views:
                    cosine = candidate_cosines(vectors[view], query_vectors[j], selected)
                    weighted = np.maximum(cosine-.6, 0)*(.2+.8*geo)*category
                    features = np.column_stack([features, cosine, weighted]).astype(np.float32)
                names = feature_names(engine, views)
                labels = np.isin(selected, list(truth)).astype(np.uint8)
                probability = np.where((labels == 1) | np.isin(selected, hard), 1.,
                                       min(1., 120./max(len(rest), 1))).astype(np.float32)
                xs.append(features)
                ys.append(labels)
                probabilities.append(probability)
                group_parts.append(np.full(len(selected), j, dtype=np.int32))
                recalls.append(len(retrieved_truth)/len(truth))
                if (j+1) % 500 == 0:
                    print(f'Признаки: {j+1}/{len(records)}, {time.time()-started:.1f} с', flush=True)
        X, y = np.vstack(xs), np.concatenate(ys)
        groups, inclusion = np.concatenate(group_parts), np.concatenate(probabilities)
        pool_recall = np.array(recalls, dtype=np.float64)
        np.savez(feature_path, X=X, y=y, group=groups, inclusion=inclusion, pool_recall=pool_recall)
        write_json(feature_meta, {'signature': signature, 'feature_names': names})

    if X.shape[1] != len(names) or (np.isinf(X).any()):
        raise ValueError('Неверная ширина признаков или бесконечные значения')
    # Неизвестный рейтинг остаётся NaN: HGB умеет обрабатывать такие пропуски.
    positive_count = np.bincount(groups, weights=y)
    weights = np.where(y == 1, 50./np.maximum(positive_count[groups], 1),
                       1./inclusion).astype(np.float32)
    model = HistGradientBoostingClassifier(**hyperparameters, early_stopping=False)
    print(f'Обучение одной конфигурации HGB: {X.shape}', flush=True)
    with threadpool_limits(limits=8):
        model.fit(X, y, sample_weight=weights)

    ranker_path = resolve(model_dir, config.get('ranker_file', 'ranker.pkl'))
    names_path = resolve(model_dir, config.get('feature_names_file', 'feature_names.json'))
    ranker_path.parent.mkdir(parents=True, exist_ok=True)
    names_path.parent.mkdir(parents=True, exist_ok=True)
    with ranker_path.open('wb') as file:
        pickle.dump(model, file, protocol=5)
    write_json(names_path, names)
    report = {**split_report, 'training_rows': len(y), 'positive_rows': int(y.sum()),
              'feature_count': X.shape[1], 'feature_names': names,
              'hyperparameters': hyperparameters, 'views': views,
              'training_pool': 'expanded', 'inference_pool': config.get('pool', 'expanded'),
              'query_mode': 'plain64',
              'sampling': {'seed': 20260930, 'hard': 70, 'random': 120,
                           'negative_weight': '1 / inclusion_probability',
                           'positive_weight': '50 / retrieved_positive_count_for_query',
                           'positive_injection_outside_pool': False},
              'training_pool_recall': float(pool_recall.mean()),
              'holdout_evaluated': False, 'encoder_revision': revision,
              'device': device, 'elapsed_seconds': time.time()-started,
              'ranker_sha256': hashlib.sha256(ranker_path.read_bytes()).hexdigest()}
    write_json(model_dir/'training_report.json', report)
    print(f'Сохранены {ranker_path} и {model_dir / "training_report.json"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', required=True, help='Каталог с исходным train.parquet')
    parser.add_argument('--model-dir', required=True, help='Каталог selected_config.json и локальных весов E5')
    parser.add_argument('--cache-dir', required=True, help='Каталог промежуточных индексов, векторов и признаков')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda'], help='Устройство E5; по умолчанию CPU')
    args = parser.parse_args()
    train_model(args.data_dir, args.model_dir, args.cache_dir, args.device)


if __name__ == '__main__':
    main()
