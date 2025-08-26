import re
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
        Logger.info("Text:", sample.text)
        Logger.info("#" * 20 + "\n")

    return preprocessed_samples
