from abc import abstractmethod

from numpy import ndarray


class TextVectorizerBase:
    @abstractmethod
    def fit_transform(self, raw_documents: list[str]) -> ndarray:
        """
        Fit the vectorizer to the raw documents and transform them into feature vectors.

        Parameters
        ----------
        raw_documents : list[str]
            A list of raw text documents to fit and transform.

        Returns
        -------
        ndarray
            The transformed feature vectors.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def transform(self, raw_documents: list[str]) -> ndarray:
        """
        Transform the raw documents into feature vectors.

        Parameters
        ----------
        raw_documents : list[str]
            A list of raw text documents to transform.

        Returns
        -------
        ndarray
            The transformed feature vectors.
        """
        raise NotImplementedError("Subclasses must implement this method")
