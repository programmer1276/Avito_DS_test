"""Локальная кандидатогенерация: BM25 + символьный TF-IDF + мягкая география.

Никаких скачиваний моделей, API и правил под конкретные query_id.
Индекс хранится в scipy sparse; большой dense query x corpus не создаётся.
"""
from functools import lru_cache
import html
import re
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem.snowball import RussianStemmer

SEARCH_COLS = ['search_query', 'search_location_id', 'search_is_delivery_search',
               'search_infm_params_text', 'search_category']
TOKEN_RE = re.compile(r'[a-zа-я0-9]+')
STEMMER = RussianStemmer()


def normalize(text):
    """Минимальная нормализация; не исправляем запросы вручную."""
    return ' '.join(TOKEN_RE.findall(html.unescape(str(text or '')).lower().replace('ё', 'е')))


@lru_cache(maxsize=300_000)
def stem(word):
    return STEMMER.stem(word) if re.search('[а-я]', word) else word


def stem_text(text):
    return ' '.join(stem(w) for w in normalize(text).split())


def top_indices(scores, k=50):
    """Детерминированный top-k, в том числе при равных оценках на границе."""
    k = min(k, len(scores))
    if not k:
        return np.array([], dtype=np.int32)
    # Находим порог top-k без полной сортировки корпуса.
    threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
    # Равные оценки на границе добираем по порядку строк — без случайности.
    above = np.flatnonzero(scores > threshold)
    equal = np.flatnonzero(scores == threshold)[:k - len(above)]
    idx = np.concatenate([above, equal])
    return idx[np.lexsort((idx, -scores[idx]))]


class BM25:
    """BM25 с положительным IDF log(1 + (N-df+0.5)/(df+0.5))."""
    def __init__(self, b=0.65, k1=1.2, max_features=250_000):
        self.b, self.k1 = b, k1
        self.vectorizer = CountVectorizer(tokenizer=str.split, token_pattern=None,
                                          lowercase=False, dtype=np.float32,
                                          max_features=max_features)

    def fit(self, texts):
        # Строка = документ, колонка = слово, ненулевое значение = его частота.
        counts = self.vectorizer.fit_transform(texts).tocsr()
        n = counts.shape[0]
        lengths = np.asarray(counts.sum(axis=1)).ravel()
        # В CSR одна позиция на слово документа: считаем документы, не повторы.
        df = np.bincount(counts.indices, minlength=counts.shape[1])
        self.idf = np.log1p((n - df + .5) / (df + .5)).astype(np.float32)
        norm = self.k1 * (1 - self.b + self.b * lengths / max(lengths.mean(), 1))
        # Для каждого ненулевого tf повторяем поправку на длину его документа.
        counts.data *= (self.k1 + 1) / (counts.data + np.repeat(norm, np.diff(counts.indptr)))
        counts.data *= self.idf[counts.indices]
        # CSC удобна для доступа к колонкам слов, присутствующих в запросе.
        self.matrix = counts.tocsc()
        return self

    def scores(self, text):
        query = self.vectorizer.transform([text]).tocsr()
        query.data[:] = 1  # Повтор слова в запросе не должен искусственно усиливать его.
        # Деление делает запросы разной длины сопоставимее, но это не вероятность.
        total_idf = float(self.idf[query.indices].sum())
        return (self.matrix @ query.T).toarray().ravel() / max(total_idf, 1e-6)


class Retriever:
    def __init__(self, items, verbose=True):
        self.items = items.reset_index(drop=True)
        self.verbose = verbose
        # Все массивы используют один порядок; в ответ возвращаем ID, не индекс.
        self.ids = self.items.item_id.to_numpy()
        self.id_to_idx = dict(zip(self.ids, range(len(self.ids))))
        self.loc = self.items.item_location_id.to_numpy()
        self.cat = self.items.item_category_id.to_numpy()
        self.lat = pd.to_numeric(self.items.item_latitude, errors='coerce').to_numpy(dtype=float)
        self.lon = pd.to_numeric(self.items.item_longitude, errors='coerce').to_numpy(dtype=float)
        coords = pd.DataFrame({'loc': self.loc, 'lat': self.lat, 'lon': self.lon})
        self.centers = coords.groupby('loc')[['lat', 'lon']].median().to_dict('index')

    def log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def fit_text(self):
        started = time.time()
        self.log(f'Индексирование {len(self.items):,} объявлений')
        title = [stem_text(x) for x in self.items.item_title_raw.fillna('')]
        self.title_index = BM25(b=.3, max_features=180_000).fit(title)
        self.log(f'  BM25 заголовков: {time.time() - started:.1f} с')
        # Ограничиваем длинные описания и повторяющиеся структурные параметры.
        # Предел одинаков для каждого документа и не зависит от разметки.
        body = (self.items.item_description_raw.fillna('').str.slice(0, 3500) + ' ' +
                self.items.item_infm_params_text.fillna('').str.slice(0, 1500))
        self.body_index = BM25().fit(stem_text(x) for x in body)
        self.log(f'  BM25 описаний: {time.time() - started:.1f} с')
        self.char_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5),
                                              min_df=2, max_features=180_000,
                                              sublinear_tf=True, dtype=np.float32)
        self.char_matrix = self.char_vectorizer.fit_transform(
            self.items.item_title_raw.fillna('').map(normalize)).tocsc()
        self.log(f'  Символьный TF-IDF: {time.time() - started:.1f} с')
        self.fit_seconds = time.time() - started
        return self

    def fit_statistics(self, train):
        """Сюда передаётся только train-fold, без dev/test текстов запросов."""
        tr = train[SEARCH_COLS + ['item_id', 'item_location_id']].copy()
        # Дубли событий не считаем независимой полезной информацией.
        tr = tr.drop_duplicates(SEARCH_COLS + ['item_id'])
        counts = tr.item_id.value_counts()
        # Популярность оставлена для сравнительного эксперимента; финальный вес 0.
        self.popularity = np.log1p(counts.reindex(self.ids, fill_value=0).to_numpy()).astype(np.float32)
        self.popularity /= max(float(self.popularity.max()), 1)
        # Для области/агломерации ID запроса может отличаться от ID города.
        # Оцениваем мягкую совместимость локаций по обучающим взаимодействиям.
        loc_counts = tr.groupby(['search_location_id', 'item_location_id']).size()
        self.loc_prior = {}
        for loc, part in loc_counts.groupby(level=0):
            values = part.droplevel(0)
            # Это сглаженная совместимость, а не вероятности с суммой 1.
            self.loc_prior[loc] = (values / (values.max() + 5)).to_dict()
        return self

    def geography(self, location):
        prior = self.loc_prior.get(location, {})
        p = pd.Series(self.loc).map(prior).fillna(0).to_numpy(dtype=np.float32)
        c = self.centers.get(location)
        if c and np.isfinite(c['lat']) and np.isfinite(c['lon']):
            # Haversine; координаты центров получены только из признаков корпуса.
            dlat = np.radians(self.lat - c['lat'])
            dlon = np.radians(self.lon - c['lon'])
            a = np.sin(dlat/2)**2 + np.cos(np.radians(self.lat))*np.cos(np.radians(c['lat']))*np.sin(dlon/2)**2
            distance = 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
            near = .8 * np.exp(-distance / 40)
            p = np.maximum(p, np.nan_to_num(near, nan=0))
        # Точное совпадение ID сильнее двух мягких географических сигналов.
        p[self.loc == location] = 1
        return p.astype(np.float32)

    def components(self, query):
        text = stem_text(query['search_query'])
        title = self.title_index.scores(text).astype(np.float32)
        body = self.body_index.scores(text).astype(np.float32)
        char = (self.char_matrix @ self.char_vectorizer.transform([normalize(query['search_query'])]).T).toarray().ravel()
        # Фильтры пока трактуются как текст: логические условия не разбираются.
        params = stem_text(query['search_infm_params_text'])
        filters = self.body_index.scores(params).astype(np.float32) if params else np.zeros(len(self.ids), dtype=np.float32)
        geo = self.geography(query['search_location_id'])
        # Ноль в категории поиска означает, что отдельного ограничения нет.
        cat = ((self.cat == query['search_category']) | (query['search_category'] == 0)).astype(np.float32)
        return dict(title=title, body=body, char=char, filters=filters, geo=geo,
                    popularity=self.popularity, category=cat)

    @staticmethod
    def score(parts, config):
        lexical = (config.get('title', .5)*parts['title'] + config.get('body', .35)*parts['body'] +
                   config.get('char', .15)*parts['char'] + config.get('filters', 0)*parts['filters'])
        # floor сохраняет шанс дальним объявлениям: жёсткого отсечения нет.
        floor = config.get('geo_floor', 1)
        geo = floor + (1-floor)*parts['geo']
        score = lexical * geo
        score *= config.get('category_floor', 1) + (1-config.get('category_floor', 1))*parts['category']
        score *= 1 + config.get('popularity', 0)*parts['popularity']
        return score

    def predict(self, queries, config, k=50):
        results = []
        start = time.time()
        for i, q in enumerate(queries.to_dict('records')):
            scores = self.score(self.components(q), config)
            results.append(self.ids[top_indices(scores, k)].tolist())
            if (i+1) % 250 == 0:
                self.log(f'  Запросы {i+1}/{len(queries)}, {time.time()-start:.1f} с')
        return results
