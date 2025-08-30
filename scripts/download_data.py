from datasets import load_dataset
from src.configuration.configuration_manager import ConfigurationManager

SETTINGS = ConfigurationManager.load()

load_dataset(
    "UniverseTBD/arxiv-abstracts-large",
    cache_dir=SETTINGS.data.external_huggingface_dir,
)
