from src.modeling.data.dataset import DatasetMetadata
from src.modeling.models.classifier.classifier_base import ClassifierBase
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from src.configuration.configuration_manager import ConfigurationManager
SETTINGS = ConfigurationManager.load()

class KmeansClassifier(ClassifierBase):
    def __init__(self, n_clusters: int):
        self._n_clusters = n_clusters

    def train_test(
        self, x_train, y_train, x_test, y_test, dataset_metadata: DatasetMetadata
    ) -> tuple[int, float, classification_report]:
        kmeans = KMeans(n_clusters=self._n_clusters, random_state=SETTINGS.random_state)
        cluster_ids = kmeans.fit_predict(x_train)

        # Assign label to clusters
        cluster_to_label = {}
        for cluster_id in set(cluster_ids):
            # Get all labels in this cluster
            labels_in_cluster = [
                y_train[i] for i in range(len(y_train)) if cluster_ids[i] == cluster_id
            ]
            most_common_label = Counter(labels_in_cluster).most_common(1)[0][0]
            cluster_to_label[cluster_id] = most_common_label

        # Predict labels for test set
        test_cluster_ids = kmeans.predict(x_test)
        y_pred = [cluster_to_label[cluster_id] for cluster_id in test_cluster_ids]
        accuracy = accuracy_score(y_test, y_pred)
        train_report = classification_report(
            y_test,
            y_pred,
            target_names=[
                dataset_metadata.id_to_label[i]
                for i in range(len(dataset_metadata.id_to_label))
            ],
            output_dict=True,
        )

        return y_pred, accuracy, train_report
