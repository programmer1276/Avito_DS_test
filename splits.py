"""Разбиение по нормализованному тексту запроса, а не по отдельным строкам."""
import numpy as np
from retrieval import SEARCH_COLS, normalize
from baseline_utils import make_splits


def make_v2_splits(train):
    # Исходные dev/test уже использовались в первой версии: новый holdout другой.
    _, old_dev, old_test = make_splits(train, n=1000, seed=42)
    texts = train.search_query.map(normalize)
    old = set(old_dev.norm_query) | set(old_test.norm_query)
    rng = np.random.default_rng(20260929)
    remaining = np.array(sorted(set(texts) - old))
    rng.shuffle(remaining)
    holdout_texts = set(remaining[:1500])
    rank_texts = set(remaining[1500:9500])
    reference = train.loc[~texts.isin(old | holdout_texts | rank_texts)]

    def choose(selected_texts):
        part = train.loc[texts.isin(selected_texts)]
        frame = part.groupby(SEARCH_COLS, sort=True, dropna=False).item_id.agg(
            lambda x: sorted(set(x))).reset_index()
        frame['_text'] = frame.search_query.map(normalize)
        frame['_random'] = rng.random(len(frame))
        return (frame.sort_values('_random').drop_duplicates('_text').sort_values('_text')
                .drop(columns=['_text', '_random']).reset_index(drop=True))

    rank_train = choose(rank_texts)
    holdout = choose(holdout_texts)
    dev = old_dev[SEARCH_COLS + ['item_id']].copy()
    assert not set(reference.search_query.map(normalize)) & (old | holdout_texts | rank_texts)
    return reference, rank_train, dev, holdout
