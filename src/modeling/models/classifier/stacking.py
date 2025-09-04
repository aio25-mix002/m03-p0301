from src.modeling.data.dataset import DatasetMetadata
from src.modeling.models.classifier.classifier_base import ClassifierBase
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from src.configuration.configuration_manager import ConfigurationManager

SETTINGS = ConfigurationManager.load()


class StackingClassifier(ClassifierBase):

    def train_test(
        self, x_train, y_train, x_test, y_test, dataset_metadata: DatasetMetadata
    ) -> tuple[int, float, classification_report]:
        sc =  StackingClassifier(
        estimators = [
            ('knn', KNeighborsClassifier(n_neighbors = 5)),
            ('rf',RandomForestClassifier(n_estimators=400,random_state=SETTINGS.random_state)),
            ('nb',GaussianNB())
        ],
        final_estimator = LogisticRegression(),
        stack_method = 'predict_proba',
        passthrough = False
        )
        sc.fit(x_train,y_train)

        # Predict on the test set
        y_pred = sc.predict(x_test)

        # Calculate accuracy and classification report
        accuracy = accuracy_score(y_test, y_pred)
        train_report = classification_report(
            y_test,
            y_pred,
            target_names=dataset_metadata.sorted_labels,
            output_dict=True,
        )

        return y_pred, accuracy, train_report
