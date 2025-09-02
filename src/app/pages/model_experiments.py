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
"""
Streamlit page for training and evaluating classification models on arXiv
abstracts.  Users can choose the type of text vectorisation (Bag‑of‑Words or
TF‑IDF), the classification algorithm (K‑Nearest Neighbours, Decision Tree,
Naive Bayes or Logistic Regression) and the method for handling class
imbalance.  The page splits the preprocessed data into training and testing
subsets, optionally performs oversampling or computes class weights, trains
the chosen model and displays the accuracy, classification report and a
confusion matrix.  Trained models and vectorisers are stored in
``st.session_state`` for reuse on the Live Prediction page.
"""

import streamlit as st
from src.configuration.configuration_manager import ConfigurationManager
import matplotlib.pyplot as plt

SETTINGS = ConfigurationManager.load()
app_state = get_app_state()
import pandas as pd
import numpy as np
from src.modeling.data import dataset_loader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
try:
    # Optional import for embedding vectoriser
    from modeling.data.embedding_vectorizer import EmbeddingVectorizer, FAISSEmbeddingVectorizer
except ImportError:
    EmbeddingVectorizer = None  # type: ignore[misc]
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    silhouette_score,
    davies_bouldin_score,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
from sklearn.cluster import KMeans

# Optional imports for advanced oversampling techniques
try:
    from imblearn.over_sampling import SMOTE, ADASYN  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    SMOTE = None  # type: ignore[assignment]
    ADASYN = None  # type: ignore[assignment]
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
import hashlib


@st.cache_data(show_spinner=True)
def load_preprocessed_dataframe(
    balanced: bool = False, *, advanced: bool = False
) -> pd.DataFrame:
    """
    Load and preprocess the dataset, returning a DataFrame with two columns:
    ``text`` and ``label``.  If ``balanced`` is True, a balanced subset of
    samples is extracted such that each category has equal representation.
    If ``advanced`` is True, the advanced preprocessing pipeline is applied
    (stopword removal, lemmatisation/stemming, rare word filtering, bigram
    detection).  Otherwise basic preprocessing is used.

    Parameters
    ----------
    balanced : bool, optional
        Whether to balance the dataset via oversampling/undersampling.
    advanced : bool, optional
        Whether to apply advanced preprocessing.  Defaults to False (basic).

    Returns
    -------
    pd.DataFrame
        A DataFrame with ``text`` and ``label`` columns.
    """
    data = dataset_loader.load_data()
    if balanced:
        raw_samples = dataset_loader.extract_balanced_samples(data)
    else:
        raw_samples = dataset_loader.extract_samples(data)
    if advanced:
        processed = dataset_loader.transform_data_advanced(raw_samples)
    else:
        processed = dataset_loader.transform_data(raw_samples)
    return pd.DataFrame([{"text": item.text, "label": item.label} for item in processed])


# -----------------------------------------------------------------------------
# Back translation data loader
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=True)
def load_backtranslated_dataframe(file_path: str) -> pd.DataFrame:
    """
    Load a back‑translated balanced dataset from a CSV file.  The CSV file must
    contain two columns: ``text`` and ``label``.  The returned DataFrame
    includes these columns for downstream processing.

    Parameters
    ----------
    file_path : str
        Path to the CSV file containing back‑translated data.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``text`` and ``label`` columns.
    """
    try:
        df_bt = pd.read_csv(file_path)
        # Ensure only the expected columns are kept
        return df_bt[["text", "label"]]
    except FileNotFoundError:
        st.error(
            f"Không tìm thấy file back translation tại {file_path}. "
            "Vui lòng tải file CSV đã xử lý vào đường dẫn này."
        )
        return pd.DataFrame(columns=["text", "label"])


def get_vectoriser(method: str):
    """Return a vectoriser instance based on the chosen method."""
    if method == "Bag‑of‑Words (BoW)":
        return CountVectorizer(max_features=5000, stop_words="english")
    elif method == "TF‑IDF":
        return TfidfVectorizer(max_features=5000, stop_words="english")
    elif method == "Embeddings (LSA)":
        # Latent Semantic Analysis (LSA) using truncated SVD on top of TF‑IDF.
        # The recommended dimensionality for LSA is around 100 components
        # according to the scikit‑learn documentation【574126161724658†L670-L704】.
        # We construct a Pipeline so that ``fit_transform`` and ``transform``
        # automatically perform both the TF‑IDF vectorisation and the SVD
        # projection.  The Pipeline behaves like a standard vectoriser.
        tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
        svd = TruncatedSVD(n_components=100, random_state=42)
        return make_pipeline(tfidf, svd)
    elif method == "Sentence Embeddings (E5)":
        # Use a pre‑trained sentence transformer to compute dense embeddings.
        # Instantiation is wrapped in a try/except because EmbeddingVectorizer
        # will raise an ImportError if `sentence_transformers` is missing.
        if EmbeddingVectorizer is None:
            return None
        try:
            return EmbeddingVectorizer()
        except ImportError:
            return None
    elif method == "Fusion (TF‑IDF + LSA)":
        # Combine TF‑IDF and LSA representations by concatenating their feature
        # vectors.  We build two pipelines: one for TF‑IDF and one for LSA (TF‑IDF
        # followed by TruncatedSVD), then use a custom wrapper to produce a
        # concatenated feature matrix.  Since scikit‑learn's FeatureUnion can
        # handle this seamlessly with sparse matrices, we use it here.  Note
        # that this representation may have higher dimensionality than either
        # individual method.
        from sklearn.pipeline import FeatureUnion

        tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
        lsa = make_pipeline(
            TfidfVectorizer(max_features=5000, stop_words="english"),
            TruncatedSVD(n_components=100, random_state=42),
        )
        return FeatureUnion([
            ("tfidf", tfidf),
            ("lsa", lsa),
        ])
    elif method == 'Embedding with FAISS index':
        @st.cache_resource
        def get_faiss_vectorizer():
            """Get cached FAISS vectorizer"""
            return FAISSEmbeddingVectorizer(
                model_name='intfloat/multilingual-e5-base',
                cache_dir="./cache/faiss",
                index_type="flat"  # or "ivf", "hnsw"
            )
            
        @st.cache_data
        def build_faiss_index(texts, mode='passage'):
            """Build and cache FAISS index"""
            vectorizer = get_faiss_vectorizer()
            return vectorizer.build_index_from_texts(texts, mode)

        vectorizer = get_faiss_vectorizer()
        
        return vectorizer
    else:
        raise ValueError(f"Unknown vectorisation method: {method}")


# -----------------------------------------------------------------------------
# Model caching
# -----------------------------------------------------------------------------
def get_model_filename(
    imb_opt: str,
    vec_opt: str,
    model_opt: str,
    params: dict,
) -> str:
    """
    Construct a deterministic filename for a trained model and vectoriser
    corresponding to the given configuration.  The filename incorporates a
    hash of the configuration to avoid collisions and preserve reproducibility.

    Parameters
    ----------
    imb_opt : str
        Selected imbalance handling strategy.
    vec_opt : str
        Selected vectorisation method.
    model_opt : str
        Selected classification model.
    params : dict
        Hyperparameters for the model.

    Returns
    -------
    str
        Absolute path to the file where the model should be saved.
    """
    # Create a unique key from configuration components
    key = f"{imb_opt}_{vec_opt}_{model_opt}_{sorted(params.items())}"
    digest = hashlib.md5(key.encode()).hexdigest()
    # Determine the artifacts directory relative to the project root
    artifacts_dir = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "artifacts")
    )
    os.makedirs(artifacts_dir, exist_ok=True)
    filename = f"model_{digest}.pkl"
    return os.path.join(artifacts_dir, filename)


def fit_vectoriser(vectoriser, train_texts, test_texts):
    """
    Fit the vectoriser on the training text and transform both training and test
    sets.  Returns (X_train, X_test).
    """
    X_train = vectoriser.fit_transform(train_texts)
    X_test = vectoriser.transform(test_texts)
    return X_train, X_test


def train_model(
    model_name: str,
    X_train,
    y_train,
    params: dict,
    class_weight: dict | str | None = None,
    sample_weight: np.ndarray | None = None,
) -> object:
    """
    Initialise and train a model according to the selected name.  The parameter
    dictionary may include hyperparameters such as ``n_neighbors`` for KNN or
    ``max_depth`` for Decision Tree.  Models are fitted on the provided
    training data.  Additional arguments ``class_weight`` and ``sample_weight``
    allow the caller to handle imbalanced datasets via cost‑sensitive learning or
    per‑sample weighting.  For MultinomialNB the input must be dense when using
    sparse matrices for certain operations.

    Returns the trained model instance.
    """
    if model_name == "K‑Nearest Neighbours (KNN)":
        n_neighbors = params.get("n_neighbors", 5)
        weighted = params.get("weighted", False)
        weight_type = "distance" if weighted else "uniform"
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weight_type)
        # KNN does not support class_weight or sample_weight during fitting.
        model.fit(X_train, y_train)
    elif model_name == "Decision Tree":
        max_depth = params.get("max_depth")
        min_samples_leaf = params.get("min_samples_leaf", 1)
        ccp_alpha = params.get("ccp_alpha", 0.0)
        model = DecisionTreeClassifier(
            random_state=42,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            ccp_alpha=ccp_alpha,
            class_weight=class_weight,
        )
        if sample_weight is not None:
            model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train)
    elif model_name == "Naive Bayes":
        # Use MultinomialNB for text data to better handle frequency counts.  This
        # classifier supports class/sample weights via the ``sample_weight`` parameter.
        model = MultinomialNB()
        dense_train = X_train.toarray() if hasattr(X_train, "toarray") else X_train
        if sample_weight is not None:
            model.fit(dense_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(dense_train, y_train)
    else:  # Logistic Regression
        model = LogisticRegression(max_iter=1000, class_weight=class_weight)
        if sample_weight is not None:
            model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train)
    return model


def evaluate_model(model, model_name: str, X_test, y_test):
    """
    Predict on the test data and compute accuracy, classification report and
    confusion matrix.  Handles dense conversion for MultinomialNB.
    """
    if model_name == "Naive Bayes" and hasattr(X_test, "toarray"):
        y_pred = model.predict(X_test.toarray())
    else:
        y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    # Compute macro F1 score
    try:
        f1 = f1_score(y_test, y_pred, average="macro")
    except Exception:
        f1 = None
    # Compute ROC‑AUC (macro) if the model exposes predict_proba
    auc = None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test if model_name != "Naive Bayes" else X_test.toarray())
            # Binarise y_test
            from sklearn.preprocessing import label_binarize
            classes = np.unique(y_test)
            y_bin = label_binarize(y_test, classes=classes)
            auc = roc_auc_score(y_bin, proba, average="macro", multi_class="ovo")
    except Exception:
        auc = None
    return y_pred, accuracy, report, cm, f1, auc


st.title("Model Experiments")
st.write("Welcome to the Model Experiments Page!")
