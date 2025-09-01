"""
Custom vectoriser for generating dense sentence embeddings via a pre‑trained
transformer model.  The class uses the `sentence_transformers` library to load
a multilingual embedding model (`intfloat/multilingual-e5-base` by default) and
provides an interface similar to scikit‑learn vectorisers with a
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

from typing import Iterable, List, Literal, Optional
import numpy as np

try:
    # Attempt to import SentenceTransformer from sentence_transformers
    from sentence_transformers import SentenceTransformer
except Exception as e:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[misc]


class EmbeddingVectorizer:
    """Vectoriser that encodes text into dense embeddings using a pre‑trained model.

    Parameters
    ----------
    model_name : str, default ``"intfloat/multilingual-e5-base"``
        Identifier of a pre‑trained model hosted on Hugging Face.  The default
        model supports many languages.
    normalize : bool, default ``True``
        Whether to return L2‑normalised embeddings.  Normalisation often
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
    ) -> None:
        if SentenceTransformer is None:
            raise ImportError(
                "sentence_transformers package is required for EmbeddingVectorizer."
            )
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize

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
            A 2‑dimensional array where each row is the embedding vector for
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

    def transform(
        self,
        texts: Iterable[str],
        mode: Literal["query", "passage", "raw"] = "query",
    ) -> np.ndarray:  # type: ignore[override]
        """Alias for :meth:`transform_numpy` to satisfy scikit‑learn interface.

        Many scikit‑learn components expect a ``transform`` method that returns
        a 2‑D array.  This method simply delegates to
        :meth:`transform_numpy`.
        """
        return self.transform_numpy(texts, mode)

    # Provide scikit‑learn–like interfaces for compatibility
    def fit(self, X: Iterable[str], y: Optional[Iterable[str]] = None) -> "EmbeddingVectorizer":
        """No‑op fit method for API compatibility.  Returns self."""
        return self

    def fit_transform(self, X: Iterable[str], y: Optional[Iterable[str]] = None) -> np.ndarray:
        """Return embeddings for the input documents.

        Since embedding models do not require fitting on task data, this is
        equivalent to calling :meth:`transform_numpy` with ``mode='passage'``.
        """
        return self.transform_numpy(X, mode="passage")

    def transform(self, texts: Iterable[str], mode: Literal["query", "passage", "raw"] = "query") -> np.ndarray:  # type: ignore[override]
        # Type ignore is used because mypy can't reconcile overriding with different return type
        return self.transform_numpy(texts, mode)