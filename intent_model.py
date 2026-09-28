"""Лёгкая supervised модель намерения запроса, без item-текстов и внешних API.

fit получает только обучающие взаимодействия. Цель — item_microcat_id.
Полученные числа — признаки совместимости с классом, не калиброванные
вероятности релевантности конкретного объявления.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB, MultinomialNB

from retrieval import SEARCH_COLS, normalize, stem_text


class QueryIntentModel:
    """TF-IDF запроса -> распределение совместимости с микрокатегориями."""

    def __init__(self, alpha=.2, classifier='multinomial', weight_power=.5,
                 word_weight=1., char_weight=1., use_filters=False):
        self.alpha = alpha
        self.classifier = classifier
        self.weight_power = weight_power
        self.word_weight = word_weight
        self.char_weight = char_weight
        self.use_filters = use_filters

    def texts(self, queries):
        if isinstance(queries, pd.DataFrame):
            records = queries.to_dict('records')
        else:
            records = queries
        values = []
        for q in records:
            text = normalize(q['search_query'])
            if self.use_filters:
                text += ' ' + normalize(q.get('search_infm_params_text', ''))
            values.append(text)
        return values

    def prepare_fit(self, fit):
        """Убираем дубли событий, агрегируем одинаковый вход и целевой класс."""
        needed = SEARCH_COLS + ['item_id', 'item_microcat_id']
        records = fit[needed].drop_duplicates(SEARCH_COLS + ['item_id']).copy()
        records['_intent_text'] = self.texts(records)
        grouped = records.groupby(['_intent_text', 'item_microcat_id'], sort=True).size().reset_index(name='n')
        self.fit_rows_ = len(records)
        self.fit_grouped_rows_ = len(grouped)
        texts = grouped['_intent_text'].tolist()
        self.word = TfidfVectorizer(tokenizer=str.split, token_pattern=None,
                                   lowercase=False, ngram_range=(1, 2),
                                   min_df=2, max_features=100_000,
                                   sublinear_tf=True, dtype=np.float32)
        self.char = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5),
                                   min_df=2, max_features=150_000,
                                   sublinear_tf=True, dtype=np.float32)
        word = self.word.fit_transform(stem_text(t) for t in texts)
        char = self.char.fit_transform(texts)
        X = sparse.hstack([word*self.word_weight, char*self.char_weight], format='csr')
        y = grouped.item_microcat_id.to_numpy()
        weights = grouped.n.to_numpy(dtype=float) ** self.weight_power
        return X, y, weights

    def fit_prepared(self, X, y, weights):
        if self.classifier == 'complement':
            self.model = ComplementNB(alpha=self.alpha)
        elif self.classifier == 'multinomial':
            self.model = MultinomialNB(alpha=self.alpha)
        else:
            raise ValueError(self.classifier)
        self.model.fit(X, y, sample_weight=weights)
        self.classes_ = self.model.classes_
        self.class_to_index_ = {int(c): i for i, c in enumerate(self.classes_)}
        return self

    def fit(self, fit):
        return self.fit_prepared(*self.prepare_fit(fit))

    def transform(self, queries):
        texts = self.texts(queries)
        word = self.word.transform(stem_text(t) for t in texts)
        char = self.char.transform(texts)
        return sparse.hstack([word*self.word_weight, char*self.char_weight], format='csr')

    def predict_proba(self, queries):
        return self.model.predict_proba(self.transform(queries))

    def candidate_features(self, query_probs, microcats):
        """Четыре колонки для кандидатов одного запроса.

        probability, probability / max_probability, log_probability,
        совпадение с самой вероятной микрокатегорией. Неизвестный класс: p=0.
        """
        idx = np.array([self.class_to_index_.get(int(c), -1) for c in microcats])
        p = np.zeros(len(idx), dtype=np.float32)
        known = idx >= 0
        p[known] = query_probs[idx[known]]
        maximum = max(float(np.max(query_probs)), 1e-12)
        return np.column_stack([p, p/maximum, np.log(np.maximum(p, 1e-12)),
                                idx == int(np.argmax(query_probs))]).astype(np.float32)
