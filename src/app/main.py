import sys
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from app import startup as Startup
import streamlit as st
from modeling.utils.logging_utils import logger as Logger
from app.routes import Routes


def main():
    # Configure the page layout
    st.set_page_config(page_title="Topic Classifier", page_icon="📊", layout="wide")

    # Set up routes & pages
    pg = Routes.build()

    # Run
    pg.run()


# Indicate that this is the main entry point.
if __name__ == "__main__":
    Logger.info("Starting the application...")
    SETTINGS = Startup.configure()
    main()
