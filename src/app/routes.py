import streamlit as st


class Routes:
    @staticmethod
    def build():
        return st.navigation(
            [
                st.Page(
                    "pages/home.py",
                    title="Home",
                    icon=":material/home:",
                    default=True,
                ),
                st.Page(
                    "pages/data_explore.py",
                    title="Data Exploration",
                    icon=":material/explore:",
                ),
                st.Page(
                    "pages/data_processing.py",
                    title="Data Processing",
                    icon=":material/explore:",
                ),
                st.Page(
                    "pages/model_experiments.py",
                    title="Experiments",
                    icon=":material/experiment:",
                ),
                st.Page(
                    "pages/live_predict.py",
                    title="Live Prediction",
                    icon=":material/bolt:",
                ),
            ],
            position="top",
            expanded=False
        )
