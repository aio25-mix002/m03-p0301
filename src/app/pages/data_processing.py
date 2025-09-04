"""
Streamlit page for loading, cleaning and visualising the arXiv abstract dataset.

This page uses helper functions from ``modeling.data.dataset_loader`` to fetch a
subset of the UniverseTBD/arxiv‑abstracts‑large dataset, apply basic text
preprocessing and display both the raw and processed data along with a simple
label distribution.  By caching the loading routine we avoid re‑downloading
the dataset on every page refresh.
"""

import streamlit as st
import pandas as pd
from src.modeling.data import dataset_loader


@st.cache_data(show_spinner=True)
def load_and_preprocess():
    """
    Load the arXiv dataset, extract 1000 single‑label samples and apply
    preprocessing (strip whitespace/newlines, remove punctuation and digits,
    lowercase).  Returns both the raw samples and a list of ``DatasetItem``s.

    This function leverages ``dataset_loader.extract_samples`` to obtain a
    representative yet potentially imbalanced subset of the data.  If you wish
    to explore a balanced subset, you can modify this call to
    ``extract_balanced_samples``.
    """
    data = dataset_loader.load_data()
    samples = dataset_loader.extract_samples(data, top_n=1000, categories_to_select=["astro-ph", "cond-mat", "cs", "math", "physics"])
    processed = dataset_loader.transform_data(samples)
    return samples, processed


st.title("Data Processing")
st.write(
    "Trang này tải và tiền xử lý dữ liệu arXiv (1000 mẫu) rồi hiển thị "
    "một số thông tin tổng quan. Nhấn nút bên dưới để bắt đầu."
)

if st.button("Tải và tiền xử lý dữ liệu"):
    samples, processed = load_and_preprocess()

    raw_df = pd.DataFrame(samples)
    # print('line 45 processed', processed)
    proc_df = pd.DataFrame(
        [{"text": item.text, "label": item.label} for item in processed[0]]
    )

    # Create interactive tabs for a more modern UI
    tab_overview, tab_processed, tab_labels, tab_topwords = st.tabs(
        ["Dữ liệu gốc", "Dữ liệu đã tiền xử lý", "Phân bố nhãn", "Top từ"]
    )

    with tab_overview:
        st.subheader("Dữ liệu gốc")
        if not raw_df.empty:
            st.dataframe(raw_df[["title", "categories", "abstract"]].head())
        else:
            st.info("Không có dữ liệu để hiển thị.")

    with tab_processed:
        st.subheader("Dữ liệu đã tiền xử lý")
        st.dataframe(proc_df.head())

    with tab_labels:
        st.subheader("Phân bố nhãn")
        label_counts = proc_df["label"].value_counts()
        st.bar_chart(label_counts)

    with tab_topwords:
        st.subheader("Top từ phổ biến")
        # Compute word frequencies across all processed texts
        import collections
        word_counter = collections.Counter()
        for text in proc_df["text"]:
            word_counter.update(text.split())
        top_words = word_counter.most_common(20)
        if top_words:
            top_df = pd.DataFrame(top_words, columns=["word", "count"])
            st.dataframe(top_df)
            st.bar_chart(top_df.set_index("word"))
        else:
            st.info("Không tìm thấy từ phổ biến.")

        # Optional: show top words per label
        selected_label = st.selectbox(
            "Chọn nhãn để xem top từ của nhãn đó", proc_df["label"].unique()
        )
        label_counter = collections.Counter()
        for text in proc_df[proc_df["label"] == selected_label]["text"]:
            label_counter.update(text.split())
        top_label_words = label_counter.most_common(10)
        if top_label_words:
            st.write(f"Top từ cho nhãn **{selected_label}**:")
            label_df = pd.DataFrame(top_label_words, columns=["word", "count"])
            st.dataframe(label_df)
            st.bar_chart(label_df.set_index("word"))
        else:
            st.info("Không tìm thấy từ phổ biến cho nhãn đã chọn.")