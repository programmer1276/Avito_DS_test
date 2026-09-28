"""Additional lexical features/sources; no query labels or external models."""
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from retrieval import stem_text


class ExtraRetrieval:
    def __init__(self, base):
        self.base = base
        self.n = len(base.ids)

    def fit(self):
        started = time.time()
        self.word_vectorizer = TfidfVectorizer(
            tokenizer=str.split, token_pattern=None, lowercase=False,
            ngram_range=(1, 2), min_df=2, max_features=250_000,
            sublinear_tf=True, dtype=np.float32)
        self.word_matrix = self.word_vectorizer.fit_transform(
            stem_text(x) for x in self.base.items.item_title_raw.fillna('')).tocsc()
        print(f'Extra title word TF-IDF fitted: {self.word_matrix.shape}, {time.time()-started:.1f}s', flush=True)
        return self

    def coverage(self, index, text):
        """Fraction of distinct query stems present in each document.

        Unknown query stems remain in denominator; an OOV word does not magically
        turn partial evidence into complete coverage. Weighted version uses max
        vocabulary IDF for those unknown stems. Nothing is fitted from labels.
        """
        words = set(text.split())
        nwords = len(words)
        count = np.zeros(self.n, dtype=np.float32)
        weighted = np.zeros(self.n, dtype=np.float32)
        columns = index.vectorizer.transform([text]).indices
        denominator = float(index.idf[columns].sum())
        denominator += (nwords - len(columns)) * float(index.idf.max())
        for column in columns:
            begin, end = index.matrix.indptr[column:column+2]
            rows = index.matrix.indices[begin:end]
            count[rows] += 1
            weighted[rows] += index.idf[column]
        return count / max(nwords, 1), weighted / max(denominator, 1e-8)

    def components(self, query):
        text = stem_text(query['search_query'])
        word = (self.word_matrix @ self.word_vectorizer.transform([text]).T).toarray().ravel()
        title_cov, title_idf_cov = self.coverage(self.base.title_index, text)
        body_cov, body_idf_cov = self.coverage(self.base.body_index, text)
        return dict(word_title=word, title_coverage=title_cov, body_coverage=body_cov,
                    title_idf_coverage=title_idf_cov, body_idf_coverage=body_idf_cov)

    @staticmethod
    def sources(parts, extra, config, baseline_score):
        geo = config.get('geo_floor', .2) + (1-config.get('geo_floor', .2))*parts['geo']
        cat = config.get('category_floor', .3) + (1-config.get('category_floor', .3))*parts['category']
        lexical = (config['title']*parts['title'] + config['body']*parts['body'] +
                   config['char']*parts['char'] + config.get('filters',0)*parts['filters'])
        tc, bc = extra['title_coverage'], extra['body_coverage']
        return {
            'baseline': baseline_score,
            'word_title': extra['word_title']*geo*cat,
            'word_blend': (.35*parts['title']+.3*parts['body']+.1*parts['char']+
                           .25*extra['word_title']+config.get('filters',0)*parts['filters'])*geo*cat,
            'coverage_body': baseline_score*(.2+.8*bc**2),
            'coverage_title': baseline_score*(.4+.6*tc**2),
            'coverage_both': baseline_score*(.3+.35*tc**2+.35*bc**2),
            'coverage_idf': baseline_score*(.2+.8*extra['body_idf_coverage']**2),
            'coverage_additive': (lexical+.3*tc**2+.3*bc**2)*geo*cat,
            'geo_005': lexical*(.05+.95*parts['geo'])*cat,
            'geo_050': lexical*(.5+.5*parts['geo'])*cat,
            'lexical': lexical*cat,
        }
