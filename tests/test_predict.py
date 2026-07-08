"""Integration tests for prediction — skipped if model files are absent."""
import pytest
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from credit_risk.features import engineer_features  # noqa: E402

MODELS_DIR = Path(__file__).parent.parent / "models"
models_available = (MODELS_DIR / "xgboost_model.pkl").exists()
skip_no_models = pytest.mark.skipif(
    not models_available, reason="model files not present (run training notebooks first)"
)


def _sample_features():
    raw = {
        "RevolvingUtilizationOfUnsecuredLines":  0.5,
        "age":                                    45.0,
        "NumberOfTime30-59DaysPastDueNotWorse":   0.0,
        "DebtRatio":                              0.35,
        "MonthlyIncome":                          5000.0,
        "NumberOfOpenCreditLinesAndLoans":        8.0,
        "NumberOfTimes90DaysLate":               0.0,
        "NumberRealEstateLoansOrLines":          1.0,
        "NumberOfTime60-89DaysPastDueNotWorse":  0.0,
        "NumberOfDependents":                     2.0,
    }
    return engineer_features(raw)


@skip_no_models
def test_predict_single_xgb_probability_range():
    from credit_risk.predict import predict_single
    result = predict_single(_sample_features(), model="xgboost")
    assert 0.0 <= result["probability"] <= 1.0


@skip_no_models
def test_predict_single_xgb_keys():
    from credit_risk.predict import predict_single
    result = predict_single(_sample_features(), model="xgboost")
    assert set(result.keys()) >= {"probability", "prediction", "risk_level",
                                   "threshold", "shap_values", "bias"}


@skip_no_models
def test_predict_single_xgb_risk_level():
    from credit_risk.predict import predict_single
    result = predict_single(_sample_features(), model="xgboost")
    assert result["risk_level"] in {"Low", "Medium", "High"}


@skip_no_models
def test_predict_single_xgb_prediction_binary():
    from credit_risk.predict import predict_single
    result = predict_single(_sample_features(), model="xgboost")
    assert result["prediction"] in {0, 1}


@skip_no_models
def test_predict_single_shap_feature_count():
    from credit_risk.predict import predict_single
    result = predict_single(_sample_features(), model="xgboost")
    assert len(result["shap_values"]) == 35


@skip_no_models
def test_predict_batch_output_shape():
    from credit_risk.predict import predict_batch
    df5 = pd.concat([_sample_features()] * 5, ignore_index=True)
    result = predict_batch(df5, model="xgboost")
    assert len(result) == 5
    assert "default_probability" in result.columns
    assert "prediction"          in result.columns
    assert "risk_level"          in result.columns


@skip_no_models
def test_predict_batch_probabilities_valid():
    from credit_risk.predict import predict_batch
    df5 = pd.concat([_sample_features()] * 5, ignore_index=True)
    result = predict_batch(df5, model="xgboost")
    assert (result["default_probability"] >= 0).all()
    assert (result["default_probability"] <= 1).all()


@skip_no_models
def test_predict_single_lr():
    from credit_risk.predict import predict_single
    result = predict_single(_sample_features(), model="lr")
    assert 0.0 <= result["probability"] <= 1.0


@skip_no_models
def test_high_risk_customer():
    from credit_risk.predict import predict_single
    raw = {
        "RevolvingUtilizationOfUnsecuredLines":  0.99,
        "age":                                    25.0,
        "NumberOfTime30-59DaysPastDueNotWorse":   5.0,
        "DebtRatio":                              2.5,
        "MonthlyIncome":                          1500.0,
        "NumberOfOpenCreditLinesAndLoans":        3.0,
        "NumberOfTimes90DaysLate":               3.0,
        "NumberRealEstateLoansOrLines":          0.0,
        "NumberOfTime60-89DaysPastDueNotWorse":  4.0,
        "NumberOfDependents":                     3.0,
    }
    result = predict_single(engineer_features(raw), model="xgboost")
    assert result["probability"] > 0.3, "High-risk customer should have elevated probability"
