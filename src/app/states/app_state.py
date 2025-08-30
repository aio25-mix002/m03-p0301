from datetime import datetime
from src.app.services.dataset_service import RawDataInformation
import streamlit as st
from dataclasses import dataclass


@dataclass
class AppState:
    sampling_updated_at: datetime
    sampling_result: list[dict]
    sampling_data_information: RawDataInformation
    # processed_dataset: list[DatasetItem]


def get_app_state() -> AppState:
    if "app_state" not in st.session_state:
        st.session_state.app_state = AppState(
            sampling_updated_at=None,
            sampling_result=None,
            sampling_data_information=None,
        )
    return st.session_state.app_state
