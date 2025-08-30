from src.modeling.data.dataset import DatasetMetadata
from src.modeling.models.classifier.classifier_base import ClassifierBase
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB

from src.configuration.configuration_manager import ConfigurationManager

SETTINGS = ConfigurationManager.load()


class GaussianNBClassifier(ClassifierBase):

    def train_test(
        self, x_train, y_train, x_test, y_test, dataset_metadata: DatasetMetadata
    ) -> tuple[int, float, classification_report]:
        nb = GaussianNB()

        # Naive Bayes requires input to be in dense format
        x_train_dense = x_train.toarray() if hasattr(x_train, "toarray") else x_train
        x_test_dense = x_test.toarray() if hasattr(x_test, "toarray") else x_test

        nb.fit(x_train_dense, y_train)

        # Predict on the test set
        y_pred = nb.predict(x_test_dense)

        # Calculate accuracy and classification report
        accuracy = accuracy_score(y_test, y_pred)
        train_report = classification_report(
            y_test,
            y_pred,
            target_names=dataset_metadata.sorted_labels,
            output_dict=True,
        )

        return y_pred, accuracy, train_report
