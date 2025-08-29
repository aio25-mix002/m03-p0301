from dataclasses import dataclass


@dataclass
class DatasetItem:
    text: str
    label: str


@dataclass
class DatasetMetadata:
    sorted_labels: list[str]
    label_to_id: dict[str, int]
    id_to_label: dict[int, str]
