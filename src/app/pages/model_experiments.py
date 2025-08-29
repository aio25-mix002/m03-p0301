import os
from numpy import ndarray
from src.modeling.models.classifier.decision_tree_classifier import (
    DecisionTreeClassifier,
)
from src.modeling.models.classifier.gaussian_nb_classifier import GaussianNBClassifier
from src.modeling.models.classifier.kmeans_classifier import KmeansClassifier
from src.modeling.models.classifier.knn_classifier import KnnClassifier
from src.modeling.visualization.plot_metrics import plot_confusion_matrix
from src.app.states.app_state import get_app_state
from src.modeling.models.text_encoder.bagofword_text_vectorizer import (
    BagOfWordTextVectorizer,
)
from src.modeling.models.text_encoder.tfidf_text_vectorizer import TfidfTextVectorizer
from src.modeling.models.text_encoder.wordembedding_encoder import (
    WordEmbeddingTextVectorizer,
)
from src.app.services import dataset_service
import streamlit as st
from src.configuration.configuration_manager import ConfigurationManager
import matplotlib.pyplot as plt

SETTINGS = ConfigurationManager.load()
app_state = get_app_state()

st.title("Model Experiments")
# IF we haven't run the sampling we should run it first
if not app_state.sampling_result:
    st.warning("Please run the sampling first.")
    st.stop()


# Init vectorizer
vectorizer_collection = [
    BagOfWordTextVectorizer(),
    TfidfTextVectorizer(),
    WordEmbeddingTextVectorizer(),
]

# Init classifier
classifier_collection = [
    KmeansClassifier(
        n_clusters=len(app_state.sampling_data_information.toplevel_topics)
    ),
    KnnClassifier(n_neighbors=5),
    DecisionTreeClassifier(),
    GaussianNBClassifier(),
]

with st.expander("Sampling Information"):
    st.write(f"Last updated at: {app_state.sampling_updated_at}")
    st.write(f"Number of samples: {len(app_state.sampling_result):,}")
    st.write(
        f"Number of top-level topics: {len(app_state.sampling_data_information.toplevel_topics):,}"
    )
    st.write(f"Number of topics: {len(app_state.sampling_data_information.topics):,}")

    for index, topicname in enumerate(
        app_state.sampling_data_information.toplevel_topics
    ):
        topic_info = app_state.sampling_data_information.toplevel_topics[topicname]
        st.write(f"[{index + 1}] {topicname}: {topic_info.sample_count} samples")

# Show run button
if st.button(
    "Run Experiments", type="primary", use_container_width=True, key="run_experiments"
):
    # [1] Preprocess data
    st.markdown("##### [1] Preprocessing")
    preprocessed_samples, preprocessed_metadata = dataset_service.transform_data(
        app_state.sampling_result
    )
    st.write("Preprocessing complete.")

    # [2] Run test split
    st.markdown("##### [2] Train/Test Split")
    x_train, x_test, y_train, y_test = dataset_service.split_dataset(
        preprocessed_samples, preprocessed_metadata
    )
    st.metric("Training samples", len(x_train))
    st.metric("Test samples", len(x_test))

    # [3] Vectorize dataset
    st.markdown("##### [3] Vectorization")
    vector_results: dict[str, dict[str, ndarray]] = (
        {}
    )  # {vectorizer_name: {split_name: ndarray}}
    for vectorizer in vectorizer_collection:
        vectorizer_name = vectorizer.__class__.__name__
        vector_results[vectorizer_name] = {}
        with st.spinner(f"Running {vectorizer_name}..."):
            # X Train
            vector_results[vectorizer_name]["x_train"] = vectorizer.fit_transform(
                x_train
            )
            st.write(
                f"Shape of {vectorizer_name} (X Train): {vector_results[vectorizer_name]['x_train'].shape}"
            )

            # X Test
            vector_results[vectorizer_name]["x_test"] = vectorizer.transform(x_test)
            st.write(
                f"Shape of {vectorizer_name} (X Test): {vector_results[vectorizer_name]['x_test'].shape}"
            )

    # [4] Train Models
    st.markdown("##### [4] Train Models")

    cm_figures: list[plt.Figure] = []
    for classifier in classifier_collection:
        classifier_name = classifier.__class__.__name__
        st.markdown(f"###### {classifier_name}...")
        for vectorizer in vectorizer_collection:
            vectorizer_name = vectorizer.__class__.__name__
            report_dir = (
                f"{SETTINGS.data.experiments_dir}/{classifier_name}_{vectorizer_name}"
            )
            os.makedirs(report_dir, exist_ok=True)

            y_pred, accuracy, train_report = classifier.train_test(
                vector_results[vectorizer_name]["x_train"],
                y_train,
                vector_results[vectorizer_name]["x_test"],
                y_test,
                preprocessed_metadata,
            )

            st.write(
                f"Accuracy for {classifier_name} with {vectorizer_name}:  {accuracy:.4f}"
            )
            fig = plot_confusion_matrix(
                y_test,
                y_pred,
                preprocessed_metadata.sorted_labels,
                f"{classifier_name} Confusion Matrix ({vectorizer_name})",
                save_path=f"{report_dir}/confusion_matrix.png",
            )
            cm_figures.append(fig)

    col1, col2, col3 = st.columns(3)
    with col1:
        for i in range(0, len(cm_figures), 3):
            st.pyplot(cm_figures[i])
    with col2:
        for i in range(1, len(cm_figures), 3):
            st.pyplot(cm_figures[i])
    with col3:
        for i in range(2, len(cm_figures), 3):
            st.pyplot(cm_figures[i])
