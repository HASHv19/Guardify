"""
TF-IDF feature construction.
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer(
    *,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 1,
    max_features: int = 5000,
) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        strip_accents="unicode",
        sublinear_tf=True,
    )

