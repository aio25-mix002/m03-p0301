from dataclasses import dataclass


@dataclass
class DatasetItem:
    text: str
    label: str
