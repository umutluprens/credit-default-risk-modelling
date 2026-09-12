from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from credit_risk.modelling import FEATURES, generate_demo_data, run_modelling


def test_demo_schema_and_outputs(tmp_path):
    frame = generate_demo_data(rows=800, seed=9)
    assert set(FEATURES).issubset(frame.columns)
    metrics = run_modelling(frame, tmp_path)
    assert set(metrics["model"]) == {"logistic_regression", "xgboost"}
    assert metrics["roc_auc"].between(0, 1).all()
    assert (tmp_path / "model_metrics.csv").exists()
    assert (tmp_path / "roc_curve.png").exists()
    assert (tmp_path / "calibration_curve.png").exists()

