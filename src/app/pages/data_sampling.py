from src.app.services import dataset_service
from src.app.states.app_state import get_app_state
import streamlit as st
from src.configuration.configuration_manager import ConfigurationManager
from datetime import datetime

SETTINGS = ConfigurationManager.load()
app_state = get_app_state()

st.title("Data Sampling")


default_top_n = SETTINGS.sampling.default_top_n
default_topics = SETTINGS.sampling.default_topics

rawdata_info = dataset_service.get_rawdata_info()

st.markdown("### Raw data information")
st.write(f"Total records: {rawdata_info.total_records:,}")
st.write(f"Total top-level topics: {len(rawdata_info.toplevel_topics):,}")
st.write(f"Total topics: {len(rawdata_info.topics):,}")

st.markdown("### Sampling configuration")
# Creating an input field to input the n_top
selected_top_n = st.number_input(
    "Select the number of top samples to train:",
    min_value=100,
    value=default_top_n,
    max_value=rawdata_info.total_records,
    step=100,
)

# Select topics from the top level topic, order by alphabet
toplevel_categories = rawdata_info.toplevel_topics
toplevel_categories = sorted(toplevel_categories)

selected_topics = st.multiselect(
    "Select top-level topics:", options=toplevel_categories, default=default_topics
)
# for each selected topic, count number of sample, and number of subtopic
for index, item in enumerate(selected_topics):
    topic_info = rawdata_info.toplevel_topics[item]
    st.write(
        f"[{index + 1}] {item}: {topic_info.sample_count:,} samples, {len(topic_info.subtopics):,} subtopics, {topic_info.max_topics_depth} levels deep max"
    )


# Show run button
if st.button(
    "Run Sampling", type="primary", use_container_width=True, key="run_sampling"
):

    samples, sample_data_information = dataset_service.extract_samples(
        selected_topics, selected_top_n
    )
    app_state.sampling_result = samples
    app_state.sampling_data_information = sample_data_information
    app_state.sampling_updated_at = datetime.now()

    st.write(f"Number of samples extracted: {len(samples):,}")
    st.write(
        f"Number of topics in the samples: {len(sample_data_information.topics):,}"
    )
    st.write(
        f"Number of top-level topics in the samples: {len(sample_data_information.toplevel_topics):,}"
    )

    for index, topicname in enumerate(selected_topics):
        if topicname not in sample_data_information.toplevel_topics:
            st.write(f"[{index + 1}] {topicname}: No samples found")
        else:
            topic_info = sample_data_information.toplevel_topics[topicname]
            st.write(
                f"[{index + 1}] {topicname}: {topic_info.sample_count:,} samples, {len(topic_info.subtopics):,} subtopics, {topic_info.max_topics_depth} levels deep max"
            )
