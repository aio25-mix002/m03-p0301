import re
import pandas as pd
from datasets import load_dataset, DatasetDict
from modeling.data.dataset import DatasetItem
from configuration.configuration_manager import ConfigurationManager
from modeling.utils.logging_utils import logger as Logger

SETTINGS = ConfigurationManager.load()


def load_data() -> DatasetDict:
    data = load_dataset(
        "UniverseTBD/arxiv-abstracts-large",
        cache_dir=SETTINGS.data.external_huggingface_dir,
    )
    return data


def extract_samples(data: DatasetDict) -> list:
    samples = []
    TOP_N = 1_000
    CATEGORIES_TO_SELECT = ["astro-ph", "cond-mat", "cs", "math", "physics"]
    for s in data["train"]:
        if len(s["categories"].split(" ")) != 1:
            continue

        cur_category = s["categories"].strip().split(".")[0]
        if cur_category not in CATEGORIES_TO_SELECT:
            continue

        samples.append(s)

        if len(samples) >= TOP_N:
            break
    Logger.info(f"Number of samples: {len(samples)}")

    for sample in samples[:3]:
        Logger.info(f"Category: {sample['categories']}")
        Logger.info(f"Abstract: {sample['abstract']}")
        Logger.info("#" * 20 + "\n")

    return samples

# New: extract a balanced set of samples from the dataset.
#
# The original ``extract_samples`` method returns up to 1 000 records
# matching a predefined list of high‑level arXiv categories (e.g. ``"cs"`` or
# ``"physics"``).  However, the resulting dataset can still be imbalanced if one
# category dominates.  Handling class imbalance at the data level is an
# effective strategy because it presents the model with roughly equal
# representation of each class, reducing bias towards the majority class.  This
# function iterates through the HuggingFace dataset and collects up to
# ``samples_per_class`` examples per category in ``CATEGORIES_TO_SELECT``.  If
# ``samples_per_class`` is not specified, the minimum available count across
# categories is used to maximise balance.  Only single‑label records are kept,
# mirroring the behaviour of ``extract_samples``.
def extract_balanced_samples(data: DatasetDict, samples_per_class: int | None = None) -> list:
    """Return a balanced subset of the dataset.

    Args:
        data: The full HuggingFace ``DatasetDict``.
        samples_per_class: Optional number of samples per category.  If ``None``,
            the smallest available count across the selected categories is used.

    Returns:
        A list of raw dataset records with roughly equal representation from each
        high‑level category in ``CATEGORIES_TO_SELECT``.
    """
    # High‑level categories to include – keep consistent with ``extract_samples``
    CATEGORIES_TO_SELECT = ["astro-ph", "cond-mat", "cs", "math", "physics"]
    # Dictionary to accumulate samples per category
    by_category: dict[str, list] = {cat: [] for cat in CATEGORIES_TO_SELECT}

    # First pass: gather all candidates (single‑label only)
    for record in data["train"]:
        # keep only abstracts with a single category
        category_tokens = record["categories"].strip().split(" ")
        if len(category_tokens) != 1:
            continue

        # extract the high‑level category (text before the period)
        high_level = category_tokens[0].split(".")[0]
        if high_level not in by_category:
            continue
        # append to corresponding list
        by_category[high_level].append(record)

    # Determine how many samples to take from each category
    if samples_per_class is None:
        # Instead of taking the minimum count (undersampling), we take the maximum count
        # across categories.  This allows us to oversample minority classes to match
        # the largest class while optionally downsampling the majority class.  If
        # you wish to use the minimum count, pass that value explicitly via
        # ``samples_per_class``.
        counts = [len(records) for records in by_category.values() if records]
        if len(counts) == 0:
            return []
        max_per_class = max(counts)
    else:
        max_per_class = samples_per_class

    import random  # local import to avoid polluting global namespace
    balanced_samples: list = []
    for cat, records in by_category.items():
        if len(records) >= max_per_class:
            # Downsample the majority class by randomly selecting max_per_class samples
            selected = random.sample(records, max_per_class)
        else:
            # Oversample the minority class by sampling with replacement
            # until we reach max_per_class samples.
            # random.choices returns a list with the desired length.
            selected = records + random.choices(records, k=max_per_class - len(records))
        balanced_samples.extend(selected)
        Logger.info(f"Selected {len(selected)} samples for category {cat}")

    Logger.info(f"Total balanced samples: {len(balanced_samples)}")
    return balanced_samples


def transform_data(data):
    preprocessed_samples = []
    for s in data:
        abstract = s["abstract"]
        # Remove \n characters in the middle and leading/trailing spaces
        abstract = abstract.strip().replace("\n", " ")

        # Remove special characters
        abstract = re.sub(r"[^\w\s]", "", abstract)

        # Remove digits
        abstract = re.sub(r"\d+", "", abstract)

        # Remove extra spaces
        abstract = re.sub(r"\s+", " ", abstract).strip()

        # Convert to lower case
        abstract = abstract.lower()

        # for the label, we only keep the first part
        parts = s["categories"].split(" ")
        category = parts[0].split(".")[0]

        preprocessed_samples.append(DatasetItem(text=abstract, label=category))

    # print first 3 preprocessed samples
    for sample in preprocessed_samples[:3]:
        Logger.info(f"Label: {sample.label}")
        Logger.info(f"Text: {sample.text}")
        Logger.info("#" * 20 + "\n")

    return preprocessed_samples


def load_augmented_back_translation(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load pre‑split augmented training set and untouched test set for back translation.

    This helper is used when back translation is performed externally.  The
    function expects two CSV files: one containing the augmented training
    samples and one containing the original test samples.  Each CSV must
    contain at least ``text`` and ``label`` columns.

    Parameters
    ----------
    train_path : str
        Path to the CSV file with the augmented training data.
    test_path : str
        Path to the CSV file with the untouched test data.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Two DataFrames: the first for training, the second for testing.  Both
        contain ``text`` and ``label`` columns.
    """
    import pandas as pd  # Local import to avoid heavy dependency on pandas at module import time
    train_df = pd.read_csv(train_path)[["text", "label"]]
    test_df = pd.read_csv(test_path)[["text", "label"]]
    return train_df, test_df


# -----------------------------------------------------------------------------
# Advanced text preprocessing
#
# To improve the quality of the vector representations, you can use the
# following function which performs additional NLP steps beyond the basic
# cleaning implemented in ``transform_data``.  These steps include:
#
# * Removing stopwords using NLTK's stopword list.
# * Lemmatising tokens via WordNet if the corpus is available, otherwise
#   stemming using the Snowball stemmer as a fallback.
# * Filtering out rare words based on a frequency threshold computed across
#   the dataset (default: 2 occurrences).
# * Detecting frequent bigrams (two‑word phrases) and joining them with an
#   underscore when they appear at least ``rare_threshold`` times.  This
#   captures multi‑word expressions such as ``machine_learning``.
#
# To use this preprocessing pipeline, call ``transform_data_advanced(raw_data)``
# instead of ``transform_data`` when loading the dataset in your Streamlit
# application.
def transform_data_advanced(data, rare_threshold: int = 2):
    """Perform advanced preprocessing on the dataset.

    Parameters
    ----------
    data : Iterable
        List of raw records (typically returned by ``extract_samples`` or
        ``extract_balanced_samples``) each containing ``abstract`` and
        ``categories``.
    rare_threshold : int, optional
        Minimum frequency required for a token or bigram to be retained in the
        processed text.  Defaults to 2.

    Returns
    -------
    list[DatasetItem]
        List of processed ``DatasetItem`` objects with lemmatised/stemmed
        tokens, stopwords and rare words removed, and frequent bigrams joined.
    """
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer, SnowballStemmer
    from nltk import word_tokenize
    from collections import Counter

    # Attempt to download necessary resources silently.  If offline, the
    # downloader will fail gracefully and fall back to a minimal stopword list.
    try:
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)
    except Exception:
        pass

    # Load stopwords; fall back to a minimal set if unavailable.
    try:
        stop_words = set(stopwords.words("english"))
    except Exception:
        stop_words = {
            "a", "an", "the", "and", "or", "but", "if", "while", "with", "without",
            "of", "at", "by", "for", "to", "from", "in", "on", "that", "this", "is", "are",
        }

    # Initialise lemmatiser; if unavailable, we will fall back to stemming.
    try:
        lemmatizer = WordNetLemmatizer()
        # Test lemmatiser; may raise LookupError if WordNet is missing.
        _ = lemmatizer.lemmatize("running")
    except Exception:
        lemmatizer = None
    stemmer = SnowballStemmer("english")

    # First pass: tokenise, gather token and bigram frequencies, and record labels.
    token_freq: Counter[str] = Counter()
    bigram_freq: Counter[tuple[str, str]] = Counter()
    tokenised_docs: list[list[str]] = []
    labels: list[str] = []

    for record in data:
        # Basic cleaning: remove line breaks, punctuation, digits and lower case
        abstract = record["abstract"].strip().replace("\n", " ")
        abstract = re.sub(r"[^\w\s]", "", abstract)
        abstract = re.sub(r"\d+", "", abstract)
        abstract = re.sub(r"\s+", " ", abstract).strip().lower()
        # Tokenise using NLTK
        try:
            tokens = [t for t in word_tokenize(abstract) if t.isalpha()]
        except Exception:
            tokens = abstract.split()
        tokenised_docs.append(tokens)
        # Record the label (first part of the category)
        label = record["categories"].split(" ")[0].split(".")[0]
        labels.append(label)
        # Update token frequencies excluding stopwords
        token_freq.update([t for t in tokens if t not in stop_words])
        # Update bigram frequencies
        for i in range(len(tokens) - 1):
            bigram = (tokens[i], tokens[i + 1])
            bigram_freq[bigram] += 1

    # Determine frequent bigrams based on the frequency threshold
    frequent_bigrams = {
        f"{w1}_{w2}": (w1, w2)
        for (w1, w2), count in bigram_freq.items()
        if count >= rare_threshold
    }

    processed_samples = []
    for tokens, label in zip(tokenised_docs, labels):
        # Merge frequent bigrams into a single token with an underscore
        i = 0
        merged_tokens: list[str] = []
        while i < len(tokens):
            if i < len(tokens) - 1:
                pair_key = f"{tokens[i]}_{tokens[i + 1]}"
                if pair_key in frequent_bigrams:
                    merged_tokens.append(pair_key)
                    i += 2
                    continue
            merged_tokens.append(tokens[i])
            i += 1
        # Filter out stopwords and rare tokens
        filtered_tokens = [t for t in merged_tokens if t not in stop_words and token_freq[t] >= rare_threshold]
        # Lemmatise or stem tokens
        if lemmatizer is not None:
            filtered_tokens = [lemmatizer.lemmatize(t) for t in filtered_tokens]
        else:
            filtered_tokens = [stemmer.stem(t) for t in filtered_tokens]
        # Reassemble into a single string
        processed_text = " ".join(filtered_tokens)
        processed_samples.append(DatasetItem(text=processed_text, label=label))

    # Log examples for debugging
    for sample in processed_samples[:3]:
        Logger.info(f"Advanced label: {sample.label}")
        Logger.info(f"Advanced text: {sample.text}")
        Logger.info("#" * 20 + "\n")

    return processed_samples
