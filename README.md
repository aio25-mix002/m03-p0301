# Project

- [Project](#project)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
    - [Step 1: Create the virtual environment](#step-1-create-the-virtual-environment)
    - [Step 2: Activate the virtual environment](#step-2-activate-the-virtual-environment)
    - [Step 3: Install dependencies](#step-3-install-dependencies)
  - [Preparation](#preparation)
    - [Download dataset (optional)](#download-dataset-optional)
  - [Run application](#run-application)
    - [Via CLI](#via-cli)
    - [Via Visual Studio Code Launch Profile](#via-visual-studio-code-launch-profile)
    - [Via Google Colab](#via-google-colab)
  - [Appendix](#appendix)
    - [Using UV](#using-uv)
    - [Code Linting](#code-linting)
    - [CI/CD](#cicd)


## Prerequisites
- Package Management: uv (https://docs.astral.sh/uv/getting-started/installation/)

## Setup

### Step 1: Create the virtual environment
```bash
uv venv
```

### Step 2: Activate the virtual environment

Windows:
```bash
.venv\Scripts\activate
```

### Step 3: Install dependencies
```bash
uv sync
```

## Preparation
### Download dataset (optional)
```bash
uv run ./scripts/download_data.py
```

## Run application

### Via CLI
```bash
uv run streamlit run ./src/app/main.py
```
### Via Visual Studio Code Launch Profile
Profiles:
- **Using Windows**: Debug Streamlit App (Windows - venv)

### Via Google Colab
We can run this app via Google Colab Jupyter Notebook, checkout [runbook.ipynb](./notebooks/runbook.ipynb) to get the notebook file.

Or click this link for the direct-link to Google Colab: [Open in Google Colab](https://colab.research.google.com/github/aio25-mix002/m03-p0301/blob/main/notebooks/runbook.ipynb)

## Appendix
### Using UV
**Install UV**
Ref: https://docs.astral.sh/uv/getting-started/installation/

**Restore package**
```bash
uv sync
```

**Install new package**
```bash
uv add package-name
```



### Code Linting

**Code check only**
```bash
uvx ruff check
```

**Code check and autofix**
```bash
uvx ruff check --fix
```

### CI/CD
The project includes a GitHub Actions workflow that automatically runs linting checks on:
- Pull requests to main branch
- Pushes to main branch
