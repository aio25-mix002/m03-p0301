from src.modeling.data.dataset import DatasetMetadata
from src.modeling.models.classifier.classifier_base import ClassifierBase
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier


class KnnClassifier(ClassifierBase):
    def __init__(self, n_neighbors: int):
        self._n_neighbors = n_neighbors

    def train_test(
        self, x_train, y_train, x_test, y_test, dataset_metadata: DatasetMetadata
    ) -> tuple[int, float, classification_report]:
        knn = KNeighborsClassifier(n_neighbors=self._n_neighbors)
        knn.fit(x_train, y_train)

        # Predict on the test set
        y_pred = knn.predict(x_test)

        # Calculate accuracy and classification report
        accuracy = accuracy_score(y_test, y_pred)
        train_report = classification_report(
            y_test,
            y_pred,
            target_names=dataset_metadata.sorted_labels,
            output_dict=True,
        )

        return y_pred, accuracy, train_report
