# California Housing Project

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](ca://s?q=Python_version_used_in_project)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](ca://s?q=MIT_License_explained)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](ca://s?q=Build_status_badge_explained)

## Overview
This repository contains an end-to-end Data Science project using the California Housing dataset.  
It demonstrates industry-grade practices including modular code organization, reproducibility, testing, and model interpretability.

## Objectives
- Perform exploratory data analysis (EDA) to understand the dataset
- Engineer meaningful features to improve predictive performance
- Train and evaluate regression models with reproducible pipelines
- Document results with clear reports and artifacts

## Repository Structure
```
California-Housing/
├── configs/              # Configuration files (YAML/JSON)
├── data/                 # Raw and processed datasets
├── logs/                 # Data acquisition, feature engineering, and modeling logs
├── models/               # Saved model artifacts (ignored in Git, .gitkeep tracked)
├── notebooks/            # Exploratory Jupyter notebooks
├── reports/              # Evaluation figures and SHAP interpretability plots
├── src/                  # Modular source code
│   ├── data/             # Data ingestion and preprocessing
│   ├── features/         # Feature engineering
│   ├── models/           # Model training
│   ├── evaluation/       # Model evaluation and visualization
│   └── utils/            # Config loader and utilities
├── tests/                # Unit tests
├── requirements.txt      # Python dependencies
├── pyproject.toml        # Project metadata
├── LICENSE               # MIT License
└── README.md             # Project documentation
```


## Key Features
- Modular architecture for data preprocessing, feature engineering, model training, and evaluation
- Config-driven pipelines for reproducibility across environments
- Logging integrated for transparency and debugging
- Unit tests for reliability
- Jupyter notebooks for exploratory analysis
- SHAP-based interpretability for model transparency

## Setup Instructions
### 1. Clone the repository
```bash
git clone git@github.com:iswastik3k/California-Housing.git
cd California-Housing
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
Project paths and parameters are stored in configs/.
Update data_config.yaml to point to your dataset locations.

## Workflow
- Place raw dataset in data/raw/.
- Run preprocessing (src/data/make_dataset.py) to generate data/processed/.
- Perform EDA in notebooks/01-eda.ipynb.
- Train models via src/models/train_model.py.
- Evaluate models via src/evaluation/evaluate_model.py.
- Logs are saved in logs/, figures and SHAP plots in reports/.

## Model Evaluation
The tuned XGBoost model achieves:

- RMSE: ~26,315
- MAE: ~16,624
- R²: ~0.95

Artifacts produced (saved in reports/):

- reports/residuals.png → residuals vs predictions
- reports/feature_importance.png → split-based feature importance
- reports/shap_importance.png → SHAP mean(|value|) bar plot
- reports/shap_summary.png → SHAP beeswarm plot showing feature impact

### Residuals vs Predictions
Shows error distribution across predicted values. Residuals are centered around 0, with higher variance for expensive homes.

### Feature Importance
XGBoost split-based importance highlights dominant features like `median_income`.

### SHAP Interpretability
SHAP plots provide a more reliable view:
- `median_income`, `latitude`, `longitude` are the strongest drivers.
- Higher income increases predicted house value.
- Inland proximity lowers predicted house value.

## Reports
Evaluation artifacts are saved in the `reports/` directory:

- [Residuals Plot](reports/residuals.png) — error distribution across predictions
- [Feature Importance](reports/feature_importance.png) — split-based importance
- [SHAP Importance](reports/shap_importance.png) — mean(|SHAP value|) bar plot
- [SHAP Summary](reports/shap_summary.png) — beeswarm plot showing feature impact

## How to Reproduce Results
0. **Setup & Configuration** ([Go to Setup](#setup-instructions))
    <br>
    <br>
1. **Preprocess data**
   ```bash
   python -m src.data.make_dataset
   ```
2. **Train model**
    ```bash
    python -m src.models.train_model
    ```
3. **Evaluate model**
    ```bash
    python -m src.evaluation.evaluate_model
    ```
Artifacts will be saved in models/, logs in logs/, and plots in reports/.


## License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
