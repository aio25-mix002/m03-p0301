from dataclasses import dataclass
from src.modeling.data import dataset_loader
from datasets import DatasetDict
from src.modeling.data.dataset import DatasetItem, DatasetMetadata
import streamlit as st


@dataclass
class SubtopicInformation:
    name: str
    sample_count: int


@dataclass
class TopicInformation:
    name: str
    sample_count: int
    subtopics: dict[str, SubtopicInformation]
    max_topics_depth: int


@dataclass
class RawDataInformation:
    total_records: int
    topics: dict[str, TopicInformation]
    toplevel_topics: dict[str, TopicInformation]
    max_topic_depth: int


def _update_topic_info(result, topic: str):
    topic_parts = topic.split(".")
    result.max_topic_depth = max(result.max_topic_depth, len(topic_parts))

    # Flat level topics
    if topic not in result.topics:
        result.topics[topic] = TopicInformation(
            name=topic,
            sample_count=1,
            subtopics={},
            max_topics_depth=0,
        )

    result.topics[topic].sample_count += 1

    # Update the topic depth if it larger than the current depth
    result.topics[topic].max_topics_depth = max(
        result.topics[topic].max_topics_depth, len(topic_parts)
    )

    # Get top-level topic
    top_topic = topic_parts[0]
    if top_topic not in result.toplevel_topics:
        result.toplevel_topics[top_topic] = TopicInformation(
            name=top_topic,
            sample_count=1,
            subtopics={},
            max_topics_depth=0,
        )

    result.toplevel_topics[top_topic].sample_count += 1
    result.toplevel_topics[top_topic].max_topics_depth = max(
        result.toplevel_topics[top_topic].max_topics_depth, len(topic_parts)
    )

    # Get sub-topics
    for sub_topic in topic_parts[1:]:
        if sub_topic not in result.toplevel_topics[top_topic].subtopics:
            result.toplevel_topics[top_topic].subtopics[sub_topic] = (
                SubtopicInformation(
                    name=sub_topic,
                    sample_count=0,
                )
            )
        result.toplevel_topics[top_topic].subtopics[sub_topic].sample_count += 1


@st.cache_resource
def get_rawdata_info() -> RawDataInformation:
    raw_data = get_rawdata()
    result = RawDataInformation(
        total_records=len(raw_data["train"]),
        topics={},
        toplevel_topics={},
        max_topic_depth=0,
    )
    for category in raw_data["train"]["categories"]:
        topics = category.split(" ")
        for topic in topics:
            _update_topic_info(result, topic)

    return result


@st.cache_resource
def get_rawdata() -> DatasetDict:
    raw_data = dataset_loader.load_data()
    return raw_data


def extract_samples(
    selected_topics: list[str], selected_top_n: int
) -> tuple[list[dict], RawDataInformation]:
    raw_data = get_rawdata()
    samples = dataset_loader.extract_samples(
        raw_data, top_n=selected_top_n, categories_to_select=selected_topics
    )
    data_information = RawDataInformation(
        total_records=len(samples),
        topics={},
        toplevel_topics={},
        max_topic_depth=0,
    )
    for item in samples:
        topics = item["categories"].split(" ")
        for topic in topics:
            _update_topic_info(data_information, topic)

    return samples, data_information


def transform_data(data: list[dict]) -> tuple[list[DatasetItem], DatasetMetadata]:
    preprocessed_samples, preprocessed_metadata = dataset_loader.transform_data(data)
    return preprocessed_samples, preprocessed_metadata


def split_dataset(
    dataset: list[DatasetItem], data_metadata: DatasetMetadata
) -> tuple[list[DatasetItem], list[DatasetItem], list[int], list[int]]:
    x_train, x_test, y_train, y_test = dataset_loader.split_dataset(
        dataset, data_metadata
    )
    return x_train, x_test, y_train, y_test
