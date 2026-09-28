"""Проверка CSV и воспроизведение исходного разбиения."""
import re
import numpy as np
import pandas as pd
from retrieval import SEARCH_COLS, normalize

def make_splits(train, n=1000, seed=42):
    norm = train.search_query.map(normalize)
    # Сначала сортируем: seed сам по себе не фиксирует случайный входной порядок.
    names = np.array(sorted(norm.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    assert len(names) > 2*n
    held = [set(names[:n]), set(names[n:2*n])]
    splits = []
    for texts in held:
        part = train.loc[norm.isin(texts), SEARCH_COLS+['item_id']].copy()
        # Все позитивные item_id выбранного контекста; повторные события удаляем.
        grouped = part.groupby(SEARCH_COLS, sort=True, dropna=False).item_id.agg(lambda x: sorted(set(x))).reset_index()
        grouped['norm_query'] = grouped.search_query.map(normalize)
        grouped['_random'] = rng.random(len(grouped))
        chosen = grouped.sort_values('_random').drop_duplicates('norm_query').sort_values('norm_query')
        splits.append(chosen.drop(columns=['_random']).reset_index(drop=True))
    # Убираем и другие контексты held-out текстов, не только выбранные сигнатуры.
    fit = train.loc[~norm.isin(held[0] | held[1])]
    assert not set(fit.search_query.map(normalize)) & (held[0] | held[1])
    return fit, *splits

def check_answer(answer_path, query_path, item_path):
    # Даже ID из одних цифр должны оставаться строками, включая ведущие нули.
    df = pd.read_csv(answer_path,dtype=str,keep_default_na=False,encoding='utf-8')
    queries = pd.read_parquet(query_path,columns=['query_id'])
    items = set(pd.read_parquet(item_path,columns=['item_id']).item_id)
    assert list(df.columns)==['query_id','answer'], 'Неверные колонки'
    assert len(df)==len(queries) and df.query_id.is_unique, 'Число/дубли запросов'
    assert set(df.query_id)==set(queries.query_id), 'Неверное множество query_id'
    assert df.query_id.str.len().eq(16).all(), 'Длина query_id'
    lengths=[]
    for answer in df.answer:
        ids=answer.split(' ') if answer else []
        assert 1<=len(ids)<=50 and len(ids)==len(set(ids)), 'Число/дубли item_id'
        assert all(re.fullmatch('[0-9a-f]{16}',x) for x in ids), 'Формат item_id'
        assert set(ids)<=items, 'item_id отсутствует в корпусе'
        lengths.append(len(ids))
    return dict(valid=True,rows=len(df),columns=list(df.columns),min_candidates=min(lengths),
                max_candidates=max(lengths),total_candidates=sum(lengths))
