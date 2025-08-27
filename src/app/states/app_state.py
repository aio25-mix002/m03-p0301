from modeling.data.dataset import DatasetItem
import streamlit as st
from dataclasses import dataclass


@dataclass
class AppState:
    processed_dataset: list[DatasetItem]


if "app_state" not in st.session_state:
    st.session_state.app_state = AppState()


def get_app_state() -> AppState:
    return st.session_state.app_state
