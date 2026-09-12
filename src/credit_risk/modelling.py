"""Leakage-free credit default risk modelling workflow."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    average_precision_score,
    brier_score_loss,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier


TARGET = "serious_dlqin_2yrs"
FEATURES = [
    "revolving_utilization",
    "age",
    "past_due_30_59",
    "debt_ratio",
    "monthly_income",
    "open_credit_lines",
    "past_due_90",
    "real_estate_loans",
    "past_due_60_89",
    "dependents",
]

ALIASES = {
    "seriousdlqin2yrs": TARGET,
    "revolvingutilizationofunsecuredlines": "revolving_utilization",
    "numberoftime30_59dayspastduenotworse": "past_due_30_59",
    "numberoftime30_59dayspastduenotworse_": "past_due_30_59",
    "debtratio": "debt_ratio",
    "monthlyincome": "monthly_income",
    "numberofopencreditlinesandloans": "open_credit_lines",
    "numberoftimes90dayslate": "past_due_90",
    "numberrealestateloansorlines": "real_estate_loans",
    "numberoftime60_89dayspastduenotworse": "past_due_60_89",
    "numberoftime60_89dayspastduenotworse_": "past_due_60_89",
    "numberofdependents": "dependents",
}


def _normalise(value: str) -> str:
    compact = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return ALIASES.get(compact.replace("_", ""), ALIASES.get(compact, compact))


def load_credit_data(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [_normalise(column) for column in frame.columns]
    unnamed = [column for column in frame.columns if column.startswith("unnamed")]
    frame = frame.drop(columns=unnamed, errors="ignore")
    missing = sorted({TARGET, *FEATURES} - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    for column in [TARGET, *FEATURES]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=[TARGET]).copy()


def clean_credit_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply domain checks without using the target variable."""
    cleaned = frame.copy()
    cleaned.loc[cleaned["age"] <= 0, "age"] = np.nan
    cleaned.loc[cleaned["monthly_income"] <= 0, "monthly_income"] = np.nan
    for column in ["past_due_30_59", "past_due_60_89", "past_due_90"]:
        cleaned.loc[cleaned[column] >= 90, column] = np.nan
    for column in ["revolving_utilization", "debt_ratio"]:
        cleaned.loc[cleaned[column] < 0, column] = np.nan
    return cleaned


def generate_demo_data(rows: int = 4_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(47, 14, rows), 18, 95)
    income = rng.lognormal(mean=8.45, sigma=0.65, size=rows)
    utilisation = np.clip(rng.beta(1.6, 3.4, rows) * 1.4, 0, 2)
    late_30 = rng.poisson(0.35, rows)
    late_60 = rng.poisson(0.12, rows)
    late_90 = rng.poisson(0.10, rows)
    debt_ratio = np.clip(rng.lognormal(-1.0, 0.8, rows), 0, 5)
    log_odds = (
        -4.2
        + 2.2 * utilisation
        + 0.45 * late_30
        + 0.85 * late_60
        + 1.0 * late_90
        + 0.35 * debt_ratio
        - 0.018 * (age - 45)
    )
    probability = 1 / (1 + np.exp(-log_odds))
    target = rng.binomial(1, probability)
    return pd.DataFrame(
        {
            TARGET: target,
            "revolving_utilization": utilisation,
            "age": age,
            "past_due_30_59": late_30,
            "debt_ratio": debt_ratio,
            "monthly_income": income,
            "open_credit_lines": rng.poisson(8, rows),
            "past_due_90": late_90,
            "real_estate_loans": rng.poisson(1.0, rows),
            "past_due_60_89": late_60,
            "dependents": rng.poisson(0.8, rows),
        }
    )


def _preprocessor(scale: bool) -> ColumnTransformer:
    steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        steps.append(("scale", RobustScaler()))
    return ColumnTransformer([("numeric", Pipeline(steps), FEATURES)])


def _models(positive_weight: float) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("prepare", _preprocessor(scale=True)),
                ("model", LogisticRegression(max_iter=2_000, class_weight="balanced")),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("prepare", _preprocessor(scale=False)),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=350,
                        max_depth=3,
                        learning_rate=0.04,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        scale_pos_weight=positive_weight,
                        eval_metric="aucpr",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def run_modelling(
    frame: pd.DataFrame,
    output_dir: str | Path,
    threshold: float = 0.25,
    seed: int = 42,
) -> pd.DataFrame:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = clean_credit_features(frame)
    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURES],
        data[TARGET].astype(int),
        test_size=0.25,
        stratify=data[TARGET],
        random_state=seed,
    )
    positives = max(int(y_train.sum()), 1)
    positive_weight = (len(y_train) - positives) / positives

    metrics_rows: list[dict] = []
    fitted: dict[str, Pipeline] = {}
    predictions: dict[str, np.ndarray] = {}
    for name, model in _models(positive_weight).items():
        model.fit(x_train, y_train)
        probability = model.predict_proba(x_test)[:, 1]
        predicted_class = (probability >= threshold).astype(int)
        report = classification_report(y_test, predicted_class, output_dict=True, zero_division=0)
        metrics_rows.append(
            {
                "model": name,
                "roc_auc": roc_auc_score(y_test, probability),
                "average_precision": average_precision_score(y_test, probability),
                "brier_score": brier_score_loss(y_test, probability),
                "precision_default": report["1"]["precision"],
                "recall_default": report["1"]["recall"],
                "threshold": threshold,
                "test_observations": len(y_test),
            }
        )
        fitted[name] = model
        predictions[name] = probability

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, probability in predictions.items():
        RocCurveDisplay.from_predictions(
            y_test, probability, name=name.replace("_", " "), ax=ax
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_title("Credit default ROC curves")
    fig.tight_layout()
    fig.savefig(output / "roc_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, probability in predictions.items():
        CalibrationDisplay.from_predictions(y_test, probability, n_bins=10, name=name.replace("_", " "), ax=ax)
    ax.set_title("Probability calibration")
    fig.tight_layout()
    fig.savefig(output / "calibration_curve.png", dpi=180)
    plt.close(fig)

    metrics = pd.DataFrame(metrics_rows).sort_values("average_precision", ascending=False)
    metrics.to_csv(output / "model_metrics.csv", index=False)
    metadata = {
        "classification_threshold": threshold,
        "training_observations": len(y_train),
        "test_observations": len(y_test),
        "default_rate_train": float(y_train.mean()),
        "default_rate_test": float(y_test.mean()),
        "preprocessing_fitted_on_training_only": True,
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metrics
