# Credit Default Risk Modelling

A reproducible credit-risk workflow that estimates the probability of serious delinquency, compares an interpretable baseline with gradient boosting, and evaluates both discrimination and calibration.

This is a portfolio rebuild inspired by coursework using the public **Give Me Some Credit** dataset. It fixes common modelling weaknesses such as target leakage, fitting transformations before the train/test split, and relying on accuracy for an imbalanced outcome.

## Business question

How can borrower-level financial behaviour be converted into a transparent probability of serious delinquency that supports consistent credit review?

## What the project demonstrates

- Credit-risk feature engineering
- Missing-value and impossible-value handling
- Leakage-free preprocessing pipelines
- Stratified train/test validation
- Logistic-regression benchmark
- Gradient-boosted decision trees
- ROC-AUC, average precision, Brier score, precision, and recall
- Threshold-based classification reporting
- ROC and calibration charts
- Synthetic demo mode for immediate reproducibility

## Repository structure

```text
.
├── data/README.md
├── src/credit_risk/
│   ├── __init__.py
│   └── modelling.py
├── tests/test_modelling.py
├── run_model.py
└── requirements.txt
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_model.py --demo --output results
```

To use the original Kaggle data after downloading it yourself:

```bash
python run_model.py --input data/raw/cs-training.csv --output results
```

## Validation design

The outcome is highly imbalanced, so accuracy is not treated as the primary metric. The project reports:

- **ROC-AUC** for overall rank discrimination
- **Average precision** for performance on the minority default class
- **Brier score** for probability accuracy
- **Precision and recall** at a configurable operating threshold
- **Calibration curve** to compare predicted risk with observed delinquency

All imputation and scaling steps are fitted on training data only.

## Data availability

The original dataset is not redistributed. Download it from Kaggle's Give Me Some Credit competition and place `cs-training.csv` in `data/raw/`. The repository's synthetic demo follows the same schema and allows the full pipeline to run without external data.

## Limitations and responsible use

- This is an educational model, not a production credit-decision system.
- The dataset is historical and may not represent current applicants or jurisdictions.
- Model performance must be monitored for drift, calibration, and subgroup disparities.
- Credit decisions require legal, compliance, affordability, and human-review controls.
- Variables that act as proxies for protected characteristics require additional fairness review.

## Skills demonstrated

Python, pandas, scikit-learn, XGBoost, imbalanced classification, probability calibration, model validation, credit-risk analysis, and responsible modelling.

