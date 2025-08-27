from modeling.data import dataset_loader
import streamlit as st


@st.cache_resource
def get_rawdata_info():
    raw_data = dataset_loader.load_data()
    toplevel_category_set = set()
    all_category_set = set()
    # Collect unique labels
    for category in raw_data["train"]["categories"]:
        all_category_set.add(category)
        parts = category.split(" ")
        for part in parts:
            topic = part.split(".")[0]
            toplevel_category_set.add(topic)

    info = {
        "total_records": len(raw_data["train"]),
        "toplevel_categories": toplevel_category_set,
        "all_categories": all_category_set,
    }

    return info


st.title("Data Exploration")

st.markdown("### Raw Data")
with st.spinner("Loading data..."):
    raw_data_info = get_rawdata_info()

    st.metric("Total Records", f"{raw_data_info['total_records']:,}")
    st.metric("All Categories", f"{len(raw_data_info['all_categories']):,}")
    st.metric("Top-Level Categories", f"{len(raw_data_info['toplevel_categories']):,}")
