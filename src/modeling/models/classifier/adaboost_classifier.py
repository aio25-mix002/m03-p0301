from src.modeling.data.dataset import DatasetMetadata
from src.modeling.models.classifier.classifier_base import ClassifierBase
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import AdaBoostClassifier

from src.configuration.configuration_manager import ConfigurationManager

SETTINGS = ConfigurationManager.load()


class AdaBoostClassifier(ClassifierBase):

    def train_test(
        self, x_train, y_train, x_test, y_test, dataset_metadata: DatasetMetadata
    ) -> tuple[int, float, classification_report]:
        ada = AdaBoostClassifier(n_estimators=400,random_state=SETTINGS.random_state, learning_rate = 0.1)
        ada.fit(x_train, y_train)

        # Predict on the test set
        y_pred = ada.predict(x_test)

        # Calculate accuracy and classification report
        accuracy = accuracy_score(y_test, y_pred)
        train_report = classification_report(
            y_test,
            y_pred,
            target_names=dataset_metadata.sorted_labels,
            output_dict=True,
        )

        return y_pred, accuracy, train_report
