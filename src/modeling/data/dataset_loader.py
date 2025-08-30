import re
from datasets import load_dataset, DatasetDict
from src.modeling.data.dataset import DatasetItem, DatasetMetadata
from src.modeling.utils.logging_utils import logger as Logger
from sklearn.model_selection import train_test_split
from src.configuration.configuration_manager import ConfigurationManager
SETTINGS = ConfigurationManager.load()


def load_data() -> DatasetDict:
    data = load_dataset(
        "UniverseTBD/arxiv-abstracts-large",
        cache_dir=SETTINGS.data.external_huggingface_dir,
    )
    return data


def extract_samples(
    data: DatasetDict, top_n: int, categories_to_select: list[str]
) -> list:
    samples = []
    for s in data["train"]:
        if len(s["categories"].split(" ")) != 1:
            continue

        cur_category = s["categories"].strip().split(".")[0]
        if cur_category not in categories_to_select:
            continue

        samples.append(s)

        if len(samples) >= top_n:
            break
    Logger.info(f"Number of samples: {len(samples)}")

    for sample in samples[:3]:
        Logger.info(f"Category: {sample['categories']}")
        Logger.info(f"Abstract: {sample['abstract']}")
        Logger.info(f"{'#' * 20}\n")

    return samples


def _preprocess_sample(data: list[dict]) -> list[DatasetItem]:
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
        Logger.info(f"{'#' * 20}\n")

    return preprocessed_samples


def transform_data(data: list[dict]) -> tuple[list[DatasetItem], DatasetMetadata]:
    # Preprocess data
    preprocessed_samples = _preprocess_sample(data)

    # Generate metadata
    labels = set([s.label for s in preprocessed_samples])

    # Sort and print unique labels
    sorted_labels = sorted(labels)

    preprocessed_metadata = DatasetMetadata(
        sorted_labels=sorted_labels,
        label_to_id={label: i for i, label in enumerate(sorted_labels)},
        id_to_label={i: label for i, label in enumerate(sorted_labels)},
    )

    return preprocessed_samples, preprocessed_metadata


def split_dataset(
    dataset: list[DatasetItem], data_metadata: DatasetMetadata
) -> tuple[list[DatasetItem], list[DatasetItem], list[int], list[int]]:
    x_full = [sample.text for sample in dataset]
    y_full = [data_metadata.label_to_id[sample.label] for sample in dataset]

    x_train, x_test, y_train, y_test = train_test_split(
        x_full,
        y_full,
        test_size=SETTINGS.train.test_size,
        random_state=SETTINGS.random_state,
        stratify=y_full,
    )
    Logger.info(f"Training samples: {len(x_train)}")
    Logger.info(f"Test samples: {len(x_test)}")

    return x_train, x_test, y_train, y_test
