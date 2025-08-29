import numpy as np
from numpy import ndarray
from sklearn.feature_extraction.text import TfidfVectorizer

from src.modeling.models.text_encoder.text_vectorizer_base import TextVectorizerBase


class TfidfTextVectorizer(TextVectorizerBase):
    def __init__(self):
        self._vectorizer = TfidfVectorizer()

    def fit_transform(self, raw_documents: list[str]) -> ndarray:
        return np.array(self._vectorizer.fit_transform(raw_documents).toarray())

    def transform(self, raw_documents: list[str]) -> ndarray:
        return np.array(self._vectorizer.transform(raw_documents).toarray())
