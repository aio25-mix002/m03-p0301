from typing import Literal
import numpy as np
from sentence_transformers import SentenceTransformer
from src.modeling.models.text_encoder.text_vectorizer_base import TextVectorizerBase


class WordEmbeddingTextVectorizer(TextVectorizerBase):
    _model_name: str
    _normalize: bool
    _model: SentenceTransformer

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        normalize: bool = True,
        mode: Literal["query", "passage"] = "query",
    ):
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._normalize = normalize
        self._mode = mode

    def _format_inputs(
        self, texts: list[str], mode: Literal["query", "passage"]
    ) -> list[str]:
        if mode not in {"query", "passage"}:
            raise ValueError("Mode must be either 'query' or 'passage'")
        return [f"{mode}: {text.strip()}" for text in texts]

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.transform(texts)

    def transform(self, texts: list[str]) -> np.ndarray:
        if self._mode == "raw":
            inputs = texts
        else:
            inputs = self._format_inputs(texts, self._mode)

        embeddings = self._model.encode(inputs, normalize_embeddings=self._normalize)
        return np.array(embeddings.tolist())

    # def transform_numpy(
    #     self, texts, mode: Literal["query", "passage"] = "query"
    # ) -> np.ndarray:
    #     return np.array(self.transform(texts, mode=mode))
