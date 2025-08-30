from abc import abstractmethod

from sklearn.metrics import classification_report

from src.modeling.data.dataset import DatasetMetadata


class ClassifierBase:
    @abstractmethod
    def train_test(
        self, x_train, y_train, x_test, y_test, dataset_metadata: DatasetMetadata
    ) -> tuple[int, float, classification_report]:
        """
        Train the classifier on the training data and evaluate it on the test data.

        Parameters
        ----------
        x_train : ndarray
            Training data features.
        y_train : ndarray
            Training data labels.
        x_test : ndarray
            Test data features.
        y_test : ndarray
            Test data labels.

        Returns
        -------
        tuple[int, float, classification_report]
            - Predicted labels for the test set.
            - Accuracy of the model.
            - Classification report as a string.
        """
        raise NotImplementedError("Subclasses must implement this method")
