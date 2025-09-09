"""
Custom vectoriser for generating dense sentence embeddings via a pre-trained
transformer model.  The class uses the `sentence_transformers` library to load
a multilingual embedding model (`intfloat/multilingual-e5-base` by default) and
provides an interface similar to scikit-learn vectorisers with a
``transform`` method.  Because embedding models require no training on the
task data, ``fit_transform`` simply redirects to ``transform``.

If the ``sentence_transformers`` package is not installed (for example,
because internet access is restricted), this module will still import but
attempting to instantiate the ``EmbeddingVectorizer`` will result in an
``ImportError``.  Users who wish to use this vectoriser should install
`sentence-transformers` in their environment.

Example::

    from modeling.data.embedding_vectorizer import EmbeddingVectorizer

    vectoriser = EmbeddingVectorizer()
    embeddings = vectoriser.transform(["This is a test sentence"], mode="query")

"""

from __future__ import annotations

from typing import Iterable, List, Literal, Optional, Tuple
import numpy as np
import pickle
import hashlib
from pathlib import Path
import streamlit as st
import torch
from src.modeling.data.faiss_cache import FAISSCache

try:
    # Attempt to import SentenceTransformer from sentence_transformers
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[misc]

class EmbeddingCache:
    def __init__(self, cache_dir="./cache/embeddings"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, text_hash: str, model_name: str) -> Path:
        """Generate cache file path based on text hash and model"""
        safe_model_name = model_name.replace("/", "_").replace("-", "_")
        return self.cache_dir / f"{safe_model_name}_{text_hash}.pkl"

    def _hash_text(self, text: str) -> str:
        """Generate hash for text"""
        return hashlib.md5(text.encode()).hexdigest()

    def get_embedding(self, text: str, model, model_name: str):
        """Get embedding from cache or compute and cache"""
        text_hash = self._hash_text(text)
        cache_path = self._get_cache_path(text_hash, model_name)

        # Try to load from cache
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        # Compute embedding
        embedding = model.encode([text], normalize_embeddings=True)[0]

        # Cache the result
        with open(cache_path, 'wb') as f:
            pickle.dump(embedding, f)

        return embedding

def setup_gpu_acceleration():
    """Setup GPU acceleration if available"""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = 'cpu'
        print("Using CPU")

    return device

class EmbeddingVectorizer:
    """Vectoriser that encodes text into dense embeddings using a pre-trained model.

    Parameters
    ----------
    model_name : str, default ``"intfloat/multilingual-e5-base"``
        Identifier of a pre-trained model hosted on Hugging Face.  The default
        model supports many languages.
    normalize : bool, default ``True``
        Whether to return L2-normalised embeddings.  Normalisation often
        improves performance in similarity tasks.

    Notes
    -----
    Instances of this class will raise ``ImportError`` during initialisation if
    ``sentence_transformers`` is not installed.  In such cases the caller can
    catch the exception and choose a different vectoriser.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        normalize: bool = True,
        emb_cache_dir: str = "./cache/embeddings",
        batch_size: int = 32,
        fai_cache_dir: str = "./cache/faiss",
        index_type: str = "flat"
    ) -> None:
        if SentenceTransformer is None:
            raise ImportError(
                "sentence_transformers package is required for EmbeddingVectorizer."
            )
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize
        self.cache = EmbeddingCache(emb_cache_dir)
        self.batch_size = batch_size
        self.model_name = model_name
        self.device = setup_gpu_acceleration()

        # Using GPU if available
        if self.device == 'cuda':
            self.model = self.model.to(self.device)

        self.faiss_cache = FAISSCache(fai_cache_dir)
        self.index_type = index_type
        self.current_index_hash = None

    def _format_inputs(self, texts: Iterable[str], mode: Literal["query", "passage"]) -> List[str]:
        """Prefix each input string with the mode as required by some embedding models.

        Certain embedding models (like the E5 series) expect inputs to be prefaced
        with a task hint (e.g. ``"query: "`` or ``"passage: "``).  This helper
        constructs the list of suitably formatted strings.
        """
        return [f"{mode}: {text.strip()}" for text in texts]

    def transform_numpy(
        self,
        texts: Iterable[str],
        mode: Literal["query", "passage", "raw"] = "query",
    ) -> np.ndarray:
        """Return embeddings as a NumPy array.

        Parameters
        ----------
        texts : iterable of str
            The input documents or sentences to embed.
        mode : {"query", "passage", "raw"}, default ``"query"``
            Specifies how to format the inputs before encoding.  If ``"raw"``,
            inputs are passed to the model unchanged.  Otherwise each input is
            prefixed with the given mode (e.g. ``"query: <text>"``).

        Returns
        -------
        np.ndarray
            A 2-dimensional array where each row is the embedding vector for
            the corresponding input text.
        """
        if mode not in {"query", "passage", "raw"}:
            raise ValueError("Mode must be either 'query', 'passage' or 'raw'")

        if mode == "raw":
            inputs = list(texts)
        else:
            inputs = self._format_inputs(texts, mode)
        embeddings = self.model.encode(inputs, normalize_embeddings=self.normalize)
        return np.array(embeddings)

    def transform(self, texts: list, mode: str = 'query'):
        """Transform texts to embeddings with caching"""
        if mode == 'raw':
            formatted_texts = texts
        else:
            formatted_texts = [f"{mode}: {text.strip()}" for text in texts]

        # Use Streamlit caching for the entire batch
        embeddings = self._compute_embeddings_cached(formatted_texts)

        return np.array(embeddings)

    @st.cache_data
    def _compute_embeddings_cached(self, texts: list):
        """Cached computation for Streamlit"""
        return self.model.encode(texts,
                                 normalize_embeddings=self.normalize,
                                 device=self.device)

    # Provide scikit-learn–like interfaces for compatibility
    def fit(self, X: Iterable[str], y: Optional[Iterable[str]] = None) -> "EmbeddingVectorizer":
        """No-op fit method for API compatibility.  Returns self."""
        return self

    def fit_transform(self, X: Iterable[str], y: Optional[Iterable[str]] = None) -> np.ndarray:
        """Return embeddings for the input documents.

        Since embedding models do not require fitting on task data, this is
        equivalent to calling :meth:`transform_numpy` with ``mode='passage'``.
        """
        return self.transform_numpy(X, mode="passage")

    # def transform(
    #     self,
    #     texts: Iterable[str],
    #     mode: Literal["query", "passage", "raw"] = "query",
    # ) -> np.ndarray:  # type: ignore[override]
    #     """Alias for :meth:`transform_numpy` to satisfy scikit-learn interface.

    #     Many scikit-learn components expect a ``transform`` method that returns
    #     a 2-D array.  This method simply delegates to
    #     :meth:`transform_numpy`.
    #     """
    #     return self.transform_numpy(texts, mode)

    # def transform(self, texts: Iterable[str], mode: Literal["query", "passage", "raw"] = "query") -> np.ndarray:  # type: ignore[override]
    #     # Type ignore is used because mypy can't reconcile overriding with different return type
    #     return self.transform_numpy(texts, mode)


class FAISSEmbeddingVectorizer:
    def __init__(
        self,
        model_name: str = 'intfloat/multilingual-e5-base',
        cache_dir: str = "./cache/faiss",
        index_type: str = "flat"
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.faiss_cache = FAISSCache(cache_dir)
        self.index_type = index_type
        self.current_index_hash = None

    @st.cache_resource
    def get_model(_self):
        """Cache the SentenceTransformer model"""
        return SentenceTransformer(_self.model_name)

    def build_index_from_texts(self, texts: List[str], mode: str = 'passage') -> str:
        """Build FAISS index from texts"""
        # Format texts
        if mode == 'raw':
            formatted_texts = texts
        else:
            formatted_texts = [f"{mode}: {text.strip()}" for text in texts]

        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(formatted_texts, normalize_embeddings=True)

        # Build index
        self.current_index_hash = self.faiss_cache.build_index(
            embeddings, texts, self.model_name, self.index_type
        )

        return self.current_index_hash

    def load_existing_index(self, dataset_hash: str) -> str:
        """Load existing FAISS index"""
        self.current_index_hash = self.faiss_cache.load_index(dataset_hash)
        return self.current_index_hash

    def find_similar(self, query_text: str, k: int = 5) -> List[Tuple[str, float]]:
        """Find similar documents"""
        if self.current_index_hash is None:
            raise ValueError("No index loaded. Call build_index_from_texts() or load_existing_index() first.")

        return self.faiss_cache.get_similar_documents(query_text, self.model, k)

    def batch_find_similar(self, query_texts: List[str], k: int = 5) -> List[List[Tuple[str, float]]]:
        """Find similar documents for multiple queries"""
        if self.current_index_hash is None:
            raise ValueError("No index loaded.")

        results = []
        for query_text in query_texts:
            similar_docs = self.find_similar(query_text, k)
            results.append(similar_docs)

        return results
