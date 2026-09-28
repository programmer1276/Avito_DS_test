"""Local prediction pipeline for the supervised candidate generator.

All paths are supplied by CLI arguments/configuration. The encoder is loaded
from a local directory only; this module never downloads models or calls an API.
The plain ``query: <search_query>`` vector (64 tokens) is shared by both item
views. Title passages use 64 tokens; full passages use 128 tokens.
"""
import argparse
import hashlib
import json
import os
import pickle
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


ENCODER_REVISION = '614241f622f53c4eeff9890bdc4f31cfecc418b3'


def top_indices(scores, k=50):
    """Stable top-k, including ties at the boundary; same rule as retrieval.py."""
    scores = np.asarray(scores)
    k = min(int(k), len(scores))
    if k <= 0:
        return np.array([], dtype=np.int64)
    threshold = np.partition(scores, len(scores)-k)[len(scores)-k]
    above = np.flatnonzero(scores > threshold)
    equal = np.flatnonzero(scores == threshold)[:k-len(above)]
    result = np.concatenate([above, equal])
    return result[np.lexsort((result, -scores[result]))]


def candidate_pool(engine, q, titlecos, pool='expanded', fullcos=None):
    """Retrieve candidates from features only; no positives are injected."""
    old, parts, extra, sources = engine.retrieve(q)
    titlecos = np.asarray(titlecos)
    if titlecos.shape != (len(engine.base.ids),):
        raise ValueError('Title cosine must align with the complete item corpus')
    if pool == 'old':
        return old, parts, extra, sources, old
    if pool not in ['expanded', 'fullpool']:
        raise ValueError(f'Unknown candidate pool: {pool}')
    geo = parts['geo']
    # Keep the original float32 retrieval arithmetic: tiny dtype differences
    # can change a tied candidate at the top-k boundary.
    cat = .3+.7*parts['category']
    semantic_mult = titlecos*(.2+.8*geo)*cat
    semantic_log = titlecos+.03*np.log(.05+.95*geo)+.05*np.log(cat)
    arrays = [
        old, top_indices(sources['coverage_both'], 1000),
        top_indices(semantic_mult, 500), top_indices(semantic_log, 500),
    ]
    if pool == 'fullpool':
        fullcos = np.asarray(fullcos)
        if fullcos.shape != (len(engine.base.ids),):
            raise ValueError('fullpool requires full-view cosine over the complete corpus')
        arrays.extend([top_indices(fullcos*(.2+.8*geo)*cat, 300),
                       top_indices(fullcos+.03*np.log(.05+.95*geo)+.05*np.log(cat), 300)])
    idx = np.unique(np.concatenate(arrays))
    return idx, parts, extra, sources, old


def _candidate_cosine(cosine, idx, corpus_size):
    cosine = np.asarray(cosine, dtype=np.float32)
    if cosine.shape == (corpus_size,):
        return cosine[idx]
    if cosine.shape == (len(idx),):
        return cosine
    raise ValueError('Cosine vector must align with the corpus or candidate array')


def buildfeatures(engine, q, idx, parts, extra, sources, titlecos, fullcos=None):
    """Feature order: engine.build, then title pair, then optional full pair."""
    lexical = engine.build(q, idx, parts, extra, sources)
    geo = parts['geo'][idx]
    cat = np.where(parts['category'][idx] > 0, 1., .3)
    columns = [lexical]
    for cosine in [titlecos] + ([] if fullcos is None else [fullcos]):
        values = _candidate_cosine(cosine, idx, len(engine.base.ids))
        columns.extend([values[:, None],
                        (np.maximum(values-.6, 0)*(.2+.8*geo)*cat)[:, None]])
    return np.hstack(columns).astype(np.float32)


def feature_names(engine, views):
    names = list(engine.feature_names)
    for view in views:
        names.extend([f'semantic_{view}_cosine', f'semantic_{view}_geo'])
    return names


def score_candidates(model, X, names, baseline_log_weight=0.,
                     sampling_correction=0., old_pool_size=None):
    """Rank by model signal, with optional explicitly selected adjustments.

    Sampling correction is a legacy heuristic for the old 70+60 sampler.
    Importance-weighted final models normally use sampling_correction=0.
    Returned scores are not calibrated relevance probabilities.
    """
    if X.shape[1] != len(names) or X.shape[1] != model.n_features_in_:
        raise ValueError('Feature width differs from the trained model/schema')
    class_columns = np.flatnonzero(np.asarray(model.classes_) == 1)
    if len(class_columns) != 1:
        raise ValueError('Ranker must contain the positive label 1')
    probability = model.predict_proba(X)[:, class_columns[0]]
    if not baseline_log_weight and not sampling_correction:
        return probability
    probability = np.clip(probability, 1e-8, 1-1e-8)
    score = np.log(probability)-np.log1p(-probability)
    if baseline_log_weight:
        baseline = X[:, names.index('baseline_score')]
        score += baseline_log_weight*np.log(np.maximum(baseline, 1e-6))
    if sampling_correction:
        if old_pool_size is None:
            raise ValueError('Sampling adjustment requires the original pool size')
        ranks = np.expm1(X[:, names.index('baseline_rank')])
        inclusion = min(1., 60/max(int(old_pool_size)-70, 1))
        score += sampling_correction*np.where(ranks <= 70.00001, 0., np.log(inclusion))
    return score


def clean(text):
    return re.sub(r'\s+', ' ', str(text) if pd.notna(text) else '').strip()


def item_passages(items, view):
    title = items.item_title_raw.map(clean)
    if view == 'title':
        return ('passage: '+title).tolist()
    if view == 'full':
        return ('passage: '+title+'. '+items.item_infm_params_text.map(clean).str[:200]
                +'. '+items.item_description_raw.map(clean).str[:500]).tolist()
    raise ValueError(view)


class LocalE5Encoder:
    """Pinned local E5 helper with explicit preprocessing and deduplication."""
    def __init__(self, model_dir, device='cpu', batch_size=128,
                 revision=ENCODER_REVISION):
        path = Path(model_dir).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f'Local encoder directory missing: {path}')
        if revision != ENCODER_REVISION:
            raise ValueError('Encoder revision differs from the trained pipeline')
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        import torch
        torch.set_num_threads(4)
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(str(path), device=device, local_files_only=True)
        self.batch_size = int(batch_size)
        self.revision = revision

    def encode_texts(self, texts, max_length):
        self.model.max_seq_length = int(max_length)
        codes, unique = pd.factorize(pd.Series(texts, dtype='object'), sort=False)
        if not len(unique):
            return np.empty((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        vectors = self.model.encode(unique.tolist(), batch_size=self.batch_size,
                                    normalize_embeddings=True, convert_to_numpy=True,
                                    show_progress_bar=False).astype(np.float32)
        return vectors[codes]

    def encode_queries(self, queries):
        records = queries.to_dict('records') if isinstance(queries, pd.DataFrame) else queries
        # Deliberately no filters: one plain query vector is used for both views.
        return self.encode_texts(['query: '+clean(q['search_query']) for q in records], 64)

    def encode_items(self, items, view='title'):
        return self.encode_texts(item_passages(items, view), 64 if view == 'title' else 128)


def encode_items(items, view, model_dir, device='cpu', encoder=None):
    """Convenience wrapper; pass an encoder to reuse one loaded local model."""
    encoder = encoder or LocalE5Encoder(model_dir, device=device)
    return encoder.encode_items(items, view=view)


def encode_queries(queries, model_dir, device='cpu', encoder=None):
    """Plain query text only, 64 tokens; the same vectors serve both item views."""
    encoder = encoder or LocalE5Encoder(model_dir, device=device)
    return encoder.encode_queries(queries)


def batched_cosines(query_vectors, document_vectors, batch_size=32, blas_threads=3):
    """Yield (start_query, query-by-item cosine matrix) using bounded BLAS batches."""
    if query_vectors.shape[1] != document_vectors.shape[1]:
        raise ValueError('Query/document embedding dimensions differ')
    with threadpool_limits(limits=blas_threads):
        for begin in range(0, len(query_vectors), batch_size):
            yield begin, np.asarray(query_vectors[begin:begin+batch_size] @ document_vectors.T,
                                    dtype=np.float32)


def candidate_cosines(document_vectors, query_vector, indices):
    """Match the exact einsum used for semantic features during training.

    Retrieval uses batched matrix multiplication; feature values use einsum to
    avoid tiny summation-order differences around learned tree thresholds.
    """
    return np.einsum('ij,j->i', document_vectors[indices], query_vector)


def load_aligned_vectors(vector_path, ids_path, item_ids):
    vectors = np.load(vector_path, mmap_mode='r')
    ids = pd.read_csv(ids_path, dtype=str, keep_default_na=False).item_id.to_numpy()
    if len(ids) != len(vectors) or len(set(ids)) != len(ids):
        raise ValueError('Embedding ID mapping is inconsistent')
    if np.array_equal(ids, item_ids):
        return vectors
    positions = pd.Index(ids).get_indexer(item_ids)
    if (positions < 0).any():
        raise ValueError('Embedding cache does not cover the requested corpus')
    return np.asarray(vectors[positions], dtype=np.float32)


def _resolve(root, value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else root/path


def _source_signature(paths, configuration):
    return {'sources': [{'path': str(p.resolve()), 'size': p.stat().st_size,
                         'mtime_ns': p.stat().st_mtime_ns} for p in paths],
            'configuration': configuration}


def build_engine(items, train, intent_config):
    # The distributable keeps these modules beside this file. No workspace path
    # is silently added to sys.path; a development caller may set PYTHONPATH.
    from retrieval import Retriever
    from extra_retrieval import ExtraRetrieval
    from intent_model import QueryIntentModel
    from rerank_model import CandidateFeatures
    base = Retriever(items).fit_text().fit_statistics(train)
    extra = ExtraRetrieval(base).fit()
    intent = QueryIntentModel(**intent_config).fit(train)
    return CandidateFeatures(base, extra, intent)


def predict(data_dir, model_dir, output_dir, cache_dir, device='cpu'):
    start = time.time()
    data_dir, model_dir = Path(data_dir), Path(model_dir)
    output_dir, cache_dir = Path(output_dir), Path(cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((model_dir/'selected_config.json').read_text(encoding='utf-8'))
    views = config.get('views', ['title', 'full'])
    if views not in [['title'], ['title', 'full']]:
        raise ValueError('Supported feature views: [title] or [title, full], in that order')
    if config.get('query_mode', 'plain64') != 'plain64':
        raise ValueError('This trained pipeline requires plain query text and max_length=64')
    with _resolve(model_dir, config['ranker_file']).open('rb') as file:
        model = pickle.load(file)
    expected_names = json.loads(_resolve(model_dir, config.get('feature_names_file', 'feature_names.json')).read_text())
    item_path = data_dir/'benchmark_items.parquet'
    query_path = data_dir/'benchmark_queries.parquet'
    train_path = data_dir/'train.parquet'
    items, queries = pd.read_parquet(item_path), pd.read_parquet(query_path)
    if not items.item_id.is_unique or not queries.query_id.is_unique:
        raise ValueError('Input IDs must be unique')
    intent_config = config.get('intent', {'alpha': .2, 'use_filters': True})
    signature = _source_signature([item_path, train_path], intent_config)
    engine_path = cache_dir/'engine.pkl'
    metadata_path = cache_dir/'engine_metadata.json'
    if config.get('engine_file'):
        engine_path = _resolve(model_dir, config['engine_file'])
        with engine_path.open('rb') as file:
            engine = pickle.load(file)
    elif engine_path.exists() and metadata_path.exists() and json.loads(metadata_path.read_text()) == signature:
        with engine_path.open('rb') as file:
            engine = pickle.load(file)
    else:
        from retrieval import SEARCH_COLS
        train = pd.read_parquet(train_path, columns=SEARCH_COLS+['item_id', 'item_location_id', 'item_microcat_id'])
        engine = build_engine(items, train, intent_config)
        with engine_path.open('wb') as file:
            pickle.dump(engine, file, protocol=5)
        metadata_path.write_text(json.dumps(signature, ensure_ascii=False, indent=2))
    if not np.array_equal(engine.base.ids, items.item_id.to_numpy()):
        raise ValueError('Cached retrieval engine has a different corpus/order')
    encoder = None
    vectors = {}
    revision = config.get('encoder_revision', ENCODER_REVISION)
    required_views = list(views)
    if config.get('pool', 'expanded') == 'fullpool' and 'full' not in required_views:
        required_views.append('full')
    for view in required_views:
        external = config.get('item_vectors', {}).get(view)
        if external:
            vectors[view] = load_aligned_vectors(_resolve(model_dir, external['vectors']),
                                                 _resolve(model_dir, external['ids']), engine.base.ids)
            continue
        prefix = cache_dir/f'items_{view}'
        passages = item_passages(items, view)
        digest = hashlib.sha256('\n'.join(passages).encode('utf-8')).hexdigest()
        metadata = {'text_sha256': digest, 'view': view, 'revision': revision,
                    'max_length': 64 if view == 'title' else 128}
        meta_path = Path(str(prefix)+'.json')
        if not (Path(str(prefix)+'.npy').exists() and Path(str(prefix)+'.ids.csv').exists()
                and meta_path.exists() and json.loads(meta_path.read_text()) == metadata):
            if encoder is None:
                encoder = LocalE5Encoder(_resolve(model_dir, config['encoder_dir']), device, revision=revision)
            array = encoder.encode_items(items, view)
            np.save(str(prefix)+'.npy', array)
            items[['item_id']].to_csv(str(prefix)+'.ids.csv', index=False)
            meta_path.write_text(json.dumps(metadata, indent=2))
        vectors[view] = load_aligned_vectors(str(prefix)+'.npy', str(prefix)+'.ids.csv', engine.base.ids)
    if encoder is None:
        encoder = LocalE5Encoder(_resolve(model_dir, config['encoder_dir']), device, revision=revision)
    query_vectors = encoder.encode_queries(queries)
    records = queries.to_dict('records')
    predictions = []
    for begin, title_batch in batched_cosines(query_vectors, vectors['title'],
                                             config.get('cosine_batch_size', 32)):
        full_batch = None
        if config.get('pool', 'expanded') == 'fullpool':
            with threadpool_limits(limits=3):
                full_batch = np.asarray(query_vectors[begin:begin+len(title_batch)] @ vectors['full'].T,
                                        dtype=np.float32)
        for offset, titlecos in enumerate(title_batch):
            number = begin+offset
            q = records[number]
            idx, parts, extra, sources, old = candidate_pool(
                engine, q, titlecos, config.get('pool', 'expanded'),
                fullcos=None if full_batch is None else full_batch[offset])
            title_features = candidate_cosines(vectors['title'], query_vectors[number], idx)
            fullcos = None
            if 'full' in views:
                # Match training feature summation independently of which views
                # generated the pool; only candidate-level values are needed.
                fullcos = candidate_cosines(vectors['full'], query_vectors[number], idx)
            X = buildfeatures(engine, q, idx, parts, extra, sources, title_features, fullcos)
            actual_names = feature_names(engine, views)
            if actual_names != expected_names:
                raise ValueError(f'Feature schema mismatch: actual={actual_names}, expected={expected_names}')
            scores = score_candidates(model, X, actual_names,
                                      config.get('baseline_log_weight', 0.),
                                      config.get('sampling_correction', 0.), len(old))
            predictions.append(engine.base.ids[idx[top_indices(scores, 50)]].tolist())
            if (number+1) % 250 == 0:
                print(f'Predicted {number+1}/{len(records)}', flush=True)
    answer = pd.DataFrame({'query_id': queries.query_id,
                           'answer': [' '.join(values) for values in predictions]})
    path = output_dir/'answer.csv'
    answer.to_csv(path, index=False, encoding='utf-8')
    from baseline_utils import check_answer
    report = check_answer(path, query_path, item_path)
    assert report['min_candidates'] == report['max_candidates'] == 50
    report.update(config=config, seconds=time.time()-start,
                  sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (output_dir/'submission_check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['predict'])
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--device', default='cpu')
    args = vars(parser.parse_args())
    args.pop('command')
    predict(**args)


if __name__ == '__main__':
    main()
