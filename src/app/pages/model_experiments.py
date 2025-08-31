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
import pandas as pd
import numpy as np
from modeling.data import dataset_loader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
try:
    # Optional import for embedding vectoriser
    from modeling.data.embedding_vectorizer import EmbeddingVectorizer
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
st.write(
    "Chọn phương pháp mã hóa, phương pháp xử lý mất cân bằng và mô hình phân loại, "
    "sau đó nhấn **Huấn luyện** để đánh giá hiệu suất mô hình trên tập dữ liệu đã tiền xử lý."
)

# Select imbalance handling strategy
imbalance_option = st.selectbox(
    "Phương pháp xử lý dữ liệu mất cân bằng",
    options=[
        "Không xử lý (giữ nguyên)",
        "Cân bằng mẫu (chọn đều mỗi lớp)",
        "Oversampling ngẫu nhiên",
        "Class weighting (trọng số)"
        ,
        "SMOTE",
        "ADASYN",
        "Back translation"
    ],
    index=0,
)

# Choose preprocessing level
preprocessing_option = st.radio(
    "Phương pháp tiền xử lý",
    options=["Cơ bản", "Nâng cao"],
    index=0,
    help=(
        "Chọn 'Nâng cao' để áp dụng lọc stopwords, lemmatisation/stemming, "
        "lọc từ hiếm và phát hiện cụm từ."
    ),
)

# Load the appropriate dataset based on the imbalance handling strategy and preprocessing option.
df_train: pd.DataFrame | None = None
df_test: pd.DataFrame | None = None
if imbalance_option == "Back translation":
    # For back translation, we expect pre‑split CSV files containing the augmented
    # training set and the untouched test set.  These files should reside in the
    # ``data`` directory at the project root.  Users can adjust the paths as needed.
    train_csv = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "data",
            "arxiv_train_augmented.csv",
        )
    )
    test_csv = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "data",
            "arxiv_test_untouched.csv",
        )
    )
    # Load the pre‑split datasets.  Advanced preprocessing is not applied on these
    # files because they have already been generated externally.  If you wish
    # to apply additional cleaning, you could call transform_data_advanced here.
    df_train, df_test = dataset_loader.load_augmented_back_translation(train_csv, test_csv)
    df = None  # placeholder to signal that pre‑split frames are used
else:
    df = load_preprocessed_dataframe(
        balanced=(imbalance_option == "Cân bằng mẫu (chọn đều mỗi lớp)"),
        advanced=(preprocessing_option == "Nâng cao"),
    )
    df_train = df
    df_test = None

if (df is not None and df.empty) or (df is None and (df_train is None or df_train.empty)):
    st.warning("Không thể tải dữ liệu. Hãy kiểm tra đường dẫn đến file hoặc kết nối Internet.")
else:
    # UI controls for vectoriser and model selection
    vectoriser_option = st.selectbox(
        "Phương pháp mã hóa văn bản",
        options=[
            "Bag‑of‑Words (BoW)",
            "TF‑IDF",
            "Embeddings (LSA)",
            "Sentence Embeddings (E5)",
            "Fusion (TF‑IDF + LSA)"
        ],
        index=1,
    )
    model_option = st.selectbox(
        "Thuật toán phân loại",
        options=[
            "K‑Nearest Neighbours (KNN)",
            "Decision Tree",
            "Naive Bayes",
            "Logistic Regression",
            "K‑Means Clustering",
        ],
    )

    # Hyperparameter inputs
    params: dict[str, int | float | bool | None] = {}
    # KNN hyperparameters
    if model_option == "K‑Nearest Neighbours (KNN)":
        params["n_neighbors"] = st.slider(
            "Số láng giềng (k)", min_value=1, max_value=15, value=5
        )
        params["weighted"] = st.checkbox(
            "Sử dụng trọng số khoảng cách (Weighted KNN)", value=False
        )
    # Decision Tree hyperparameters
    elif model_option == "Decision Tree":
        params["max_depth"] = st.slider(
            "Độ sâu tối đa của cây (None = không giới hạn)", min_value=1, max_value=20, value=10
        )
        params["min_samples_leaf"] = st.slider(
            "Số mẫu tối thiểu ở lá", min_value=1, max_value=10, value=1
        )
        params["ccp_alpha"] = st.number_input(
            "Giá trị ccp_alpha (cắt tỉa); 0 = không cắt", min_value=0.0, max_value=0.05, value=0.0, step=0.005
        )
    # K‑Means hyperparameters
    elif model_option == "K‑Means Clustering":
        params["n_clusters"] = st.slider(
            "Số cụm (k)", min_value=2, max_value=10, value=5
        )

    if st.button("Huấn luyện"):
        # Determine training and testing sets.  If ``df_train`` and ``df_test`` are
        # provided (Back translation), use them directly; otherwise split ``df``.
        if df is None and df_train is not None and df_test is not None:
            X_train_text = df_train["text"]
            y_train = df_train["label"]
            X_test_text = df_test["text"]
            y_test = df_test["label"]
        else:
            X_train_text, X_test_text, y_train, y_test = train_test_split(
                df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
            )

        # If random oversampling is selected and not using pre‑split back translation,
        # oversample the training set prior to vectorisation
        if imbalance_option == "Oversampling ngẫu nhiên" and not (df is None and df_train is not None):
            train_df = pd.DataFrame({"text": X_train_text, "label": y_train}).reset_index(drop=True)
            class_counts = train_df["label"].value_counts()
            max_count = class_counts.max()
            balanced_frames = []
            for label, group in train_df.groupby("label"):
                if len(group) < max_count:
                    sampled = resample(
                        group,
                        replace=True,
                        n_samples=max_count,
                        random_state=42,
                    )
                else:
                    sampled = group
                balanced_frames.append(sampled)
            balanced_train_df = (
                pd.concat(balanced_frames)
                .sample(frac=1, random_state=42)
                .reset_index(drop=True)
            )
            X_train_text = balanced_train_df["text"]
            y_train = balanced_train_df["label"]

        # Determine the cache filename for the current configuration.  If a model
        # has already been trained with the same imbalance handling, vectorisation
        # method, classification algorithm and hyperparameters, it will be reused.
        model_file = get_model_filename(
            imbalance_option, vectoriser_option, model_option, params
        )

        # Handle the special case of K‑Means clustering separately
        if model_option == "K‑Means Clustering":
            # K‑Means is unsupervised; we cluster the entire training set (after balancing if applicable)
            vectoriser = get_vectoriser(vectoriser_option)
            if vectoriser is None:
                st.error(
                    "Không tìm thấy thư viện `sentence_transformers`. Vui lòng cài đặt\n"
                    "thư viện này hoặc chọn một phương pháp mã hóa khác."
                )
                st.stop()
            # Fit the vectoriser on all available text (train + test if back translation)
            if df is not None:
                texts_for_clustering = df["text"]
                labels_for_clustering = df["label"]
            else:
                # Use concatenated train and test sets
                texts_for_clustering = pd.concat([df_train["text"], df_test["text"]], ignore_index=True)
                labels_for_clustering = pd.concat([df_train["label"], df_test["label"]], ignore_index=True)
            X_all_vec = vectoriser.fit_transform(texts_for_clustering)
            # Determine number of clusters
            n_clusters = params.get("n_clusters", 5)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X_all_vec)
            # Compute clustering metrics if possible
            silhouette = None
            db_score = None
            try:
                if n_clusters > 1:
                    silhouette = silhouette_score(X_all_vec, cluster_labels)
                    db_score = davies_bouldin_score(X_all_vec.toarray() if hasattr(X_all_vec, "toarray") else X_all_vec, cluster_labels)
            except Exception:
                pass
            st.write(f"**Số cụm:** {n_clusters}")
            if silhouette is not None:
                st.write(f"**Silhouette Score:** {silhouette:.3f}")
            if db_score is not None:
                st.write(f"**Davies–Bouldin Index:** {db_score:.3f}")
            # Display distribution of clusters and true labels
            cluster_df = pd.DataFrame({"cluster": cluster_labels, "label": labels_for_clustering})
            st.subheader("Phân bố nhãn theo cụm")
            cluster_counts = cluster_df.groupby(["cluster", "label"]).size().unstack(fill_value=0)
            st.dataframe(cluster_counts)
            # Save clusterer and vectoriser for potential reuse (optional)
            joblib.dump({"vectorizer": vectoriser, "model": kmeans}, model_file)
            st.session_state["vectorizer"] = vectoriser
            st.session_state["model"] = kmeans
            # Persist metadata for use on the Live Prediction page.  These entries
            # allow us to display which model, vectorisation method and imbalance
            # strategy were used during training.  Without setting these values,
            # the Live Prediction page cannot provide contextual information.
            st.session_state["model_name"] = model_option
            st.session_state["vectoriser_name"] = vectoriser_option
            st.session_state["imbalance_option"] = imbalance_option
        else:
            # Classification pipeline
            # Attempt to load a cached model and vectoriser
            if os.path.exists(model_file):
                saved = joblib.load(model_file)
                vectoriser = saved["vectorizer"]
                model = saved["model"]
                st.success("Đã tải mô hình từ file cache, bỏ qua bước huấn luyện.")
                X_train_vec = vectoriser.transform(X_train_text)
                X_test_vec = vectoriser.transform(X_test_text)
            else:
                # Select vectoriser
                vectoriser = get_vectoriser(vectoriser_option)
                if vectoriser is None:
                    st.error(
                        "Không tìm thấy thư viện `sentence_transformers`. Vui lòng cài đặt\n"
                        "thư viện này hoặc chọn một phương pháp mã hóa khác."
                    )
                    st.stop()
                # Vectorise text
                X_train_vec, X_test_vec = fit_vectoriser(vectoriser, X_train_text, X_test_text)

                # Apply feature‑space oversampling methods
                if imbalance_option == "SMOTE":
                    if SMOTE is None:
                        st.error(
                            "Thuật toán SMOTE yêu cầu cài đặt thư viện imbalanced‑learn. "
                            "Vui lòng cài đặt gói `imbalanced-learn` để sử dụng tính năng này."
                        )
                        st.stop()
                    sm = SMOTE(random_state=42)
                    X_array = X_train_vec.toarray() if hasattr(X_train_vec, "toarray") else X_train_vec
                    X_array, y_train = sm.fit_resample(X_array, y_train)
                    X_train_vec = X_array
                elif imbalance_option == "ADASYN":
                    if ADASYN is None:
                        st.error(
                            "Thuật toán ADASYN yêu cầu cài đặt thư viện imbalanced‑learn. "
                            "Vui lòng cài đặt gói `imbalanced-learn` để sử dụng tính năng này."
                        )
                        st.stop()
                    ada = ADASYN(random_state=42)
                    X_array = X_train_vec.toarray() if hasattr(X_train_vec, "toarray") else X_train_vec
                    X_array, y_train = ada.fit_resample(X_array, y_train)
                    X_train_vec = X_array

                # Prepare class/sample weights
                class_weights: dict[str, float] | str | None = None
                sample_weights: np.ndarray | None = None
                if imbalance_option == "Class weighting (trọng số)":
                    unique_classes = np.unique(y_train)
                    weights = compute_class_weight(
                        class_weight="balanced", classes=unique_classes, y=y_train
                    )
                    class_weights = {cls: wt for cls, wt in zip(unique_classes, weights)}
                    sample_weights = np.array([class_weights[label] for label in y_train])

                # Train model
                model = train_model(
                    model_option,
                    X_train_vec,
                    y_train,
                    params,
                    class_weight=class_weights,
                    sample_weight=sample_weights,
                )

                # Save model and vectoriser
                joblib.dump({"vectorizer": vectoriser, "model": model}, model_file)
                st.info(f"Mô hình và vectoriser đã được lưu ở {model_file}")

            # Evaluate model
            y_pred, accuracy, report, cm, f1, auc = evaluate_model(model, model_option, X_test_vec, y_test)
            st.write(f"**Độ chính xác:** {accuracy:.2%}")
            if f1 is not None:
                st.write(f"**F1 macro:** {f1:.2f}")
            if auc is not None:
                st.write(f"**ROC‑AUC macro:** {auc:.2f}")
            # Classification report
            rep_df = pd.DataFrame(report).transpose()
            st.subheader("Báo cáo phân loại")
            st.dataframe(
                rep_df.style.format(
                    {"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support": "{:.0f}"}
                )
            )
            # Confusion matrix
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6, 4))
            labels_unique = np.unique(list(y_test))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=labels_unique,
                yticklabels=labels_unique,
                cbar=False,
                ax=ax,
            )
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            st.pyplot(fig)
            # Save objects for live prediction
            st.session_state["vectorizer"] = vectoriser
            st.session_state["model"] = model
            # Persist metadata for use on the Live Prediction page.  This
            # information is helpful for display and simple explainability.
            st.session_state["model_name"] = model_option
            st.session_state["vectoriser_name"] = vectoriser_option
            st.session_state["imbalance_option"] = imbalance_option