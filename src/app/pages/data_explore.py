from src.app.services import dataset_service
import streamlit as st
import pandas as pd

st.title("Data Exploration")

st.markdown("### Raw Data")
with st.spinner("Loading data..."):

    # Get data information
    rawdata_info = dataset_service.get_rawdata_info()

    # Display data information
    cols = st.columns(4)
    with cols[0]:
        st.metric("Total Records", f"{rawdata_info.total_records:,}")
    with cols[1]:
        st.metric("Top-Level Categories", f"{len(rawdata_info.toplevel_topics):,}")
    with cols[2]:
        st.metric("All Categories", f"{len(rawdata_info.topics):,}")
    with cols[3]:
        st.metric("Max Topic Depth", f"{rawdata_info.max_topic_depth:,}")

    # Show top-level topic information
    column_config = {
        "topic": st.column_config.TextColumn(width="medium"),
        "sample_count": st.column_config.NumberColumn(
            width="medium", format="localized"
        ),
        "topic_max_depth": st.column_config.NumberColumn(
            width="medium", format="localized"
        ),
    }
    col_1, col_2 = st.columns(2)
    with col_1:
        st.markdown("#### Top-level Topic")
        toplevel_topics = pd.DataFrame(
            [
                {
                    "topic": topicname,
                    "sample_count": info.sample_count,
                    "subtopics_count": len(info.subtopics),
                    "topic_max_depth": info.max_topics_depth,
                }
                for topicname, info in rawdata_info.toplevel_topics.items()
            ]
        )
        st.dataframe(
            toplevel_topics, column_config=column_config, use_container_width=False
        )

    # Show topic information
    with col_2:
        st.markdown("#### Topics")
        topicsdf = pd.DataFrame(
            [
                {
                    "topic": topicname,
                    "sample_count": info.sample_count,
                    "topic_max_depth": info.max_topics_depth,
                }
                for topicname, info in rawdata_info.topics.items()
            ]
        )
        st.dataframe(topicsdf, column_config=column_config, use_container_width=False)
