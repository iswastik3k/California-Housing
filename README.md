# California Housing Project

## Overview
End-to-end Data Science project using the California Housing dataset.  
Demonstrates industry-grade practices: modular code, reproducibility, testing, and deployment.

## Project Goals
- Perform exploratory data analysis (EDA)
- Engineer meaningful features
- Train and evaluate regression models
- Package and document results professionally

## Structure
- `data/` → raw, processed, external datasets
- `src/` → modular source code
- `notebooks/` → exploratory notebooks
- `reports/` → figures and summaries
- `tests/` → unit tests

## Features
- Modular architecture for data, features, models, and visualization
- Config‑driven pipelines for reproducibility
- Logging and unit testing for reliability
- Jupyter notebooks for exploratory analysis
- Reports and figures for sharing insights
- Scalable design for future extensions

## Setup
### 1. Clone the repository
```bash
git clone git@github.com:iswastik3k/California-Housing.git
cd california-housing
```
### 2. Create a virtual environment (venv)
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Configuration
Config files are stored in configs/ (YAML/JSON).

## Workflow

- Place raw dataset in data/raw/.
- Run preprocessing to generate data/processed/.
- Perform EDA in notebooks/01-eda.ipynb.
- Train models via src/models/.
- Save logs in logs/ and figures in reports/figures/.

## License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
