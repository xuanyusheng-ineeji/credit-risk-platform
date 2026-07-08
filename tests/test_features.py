"""Unit tests for feature engineering — no model files required."""
import pytest
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from credit_risk.features import engineer_features, FEATURE_ORDER, RAW_FEATURES


def _sample() -> dict:
    return {
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


# ── Schema tests ──────────────────────────────────────────────────────────────

def test_output_column_count():
    result = engineer_features(_sample())
    assert len(result.columns) == 35


def test_output_column_order():
    result = engineer_features(_sample())
    assert list(result.columns) == FEATURE_ORDER


def test_single_row_output():
    result = engineer_features(_sample())
    assert len(result) == 1


def test_batch_input():
    df = pd.DataFrame([_sample(), _sample()])
    result = engineer_features(df)
    assert len(result) == 2
    assert list(result.columns) == FEATURE_ORDER


# ── Missing-flag tests ────────────────────────────────────────────────────────

def test_missing_income_flag():
    raw = _sample()
    raw["MonthlyIncome"] = None
    result = engineer_features(raw)
    assert result["MonthlyIncome_missing"].iloc[0] == 1


def test_missing_income_fill():
    raw = _sample()
    raw["MonthlyIncome"] = None
    result = engineer_features(raw)
    assert result["MonthlyIncome"].iloc[0] == pytest.approx(5400.0)


def test_missing_dependents_flag():
    raw = _sample()
    raw["NumberOfDependents"] = None
    result = engineer_features(raw)
    assert result["NumberOfDependents_missing"].iloc[0] == 1


def test_age_zero_flag():
    raw = _sample()
    raw["age"] = 0
    result = engineer_features(raw)
    assert result["age_missing"].iloc[0] == 1


def test_age_zero_fill():
    raw = _sample()
    raw["age"] = 0
    result = engineer_features(raw)
    assert result["age"].iloc[0] == pytest.approx(52.0)


def test_late_96_flag():
    raw = _sample()
    raw["NumberOfTimes90DaysLate"] = 96
    result = engineer_features(raw)
    assert result["NumberOfTimes90DaysLate_missing"].iloc[0] == 1
    assert result["NumberOfTimes90DaysLate"].iloc[0] == pytest.approx(0.0)


def test_late_98_flag():
    raw = _sample()
    raw["NumberOfTime30-59DaysPastDueNotWorse"] = 98
    result = engineer_features(raw)
    assert result["NumberOfTime30-59DaysPastDueNotWorse_missing"].iloc[0] == 1


# ── Clipping tests ────────────────────────────────────────────────────────────

def test_utilization_clip_high():
    raw = _sample()
    raw["RevolvingUtilizationOfUnsecuredLines"] = 5.0
    result = engineer_features(raw)
    assert result["RevolvingUtilizationOfUnsecuredLines"].iloc[0] == pytest.approx(1.0)


def test_utilization_clip_low():
    raw = _sample()
    raw["RevolvingUtilizationOfUnsecuredLines"] = -0.1
    result = engineer_features(raw)
    assert result["RevolvingUtilizationOfUnsecuredLines"].iloc[0] == pytest.approx(0.0)


# ── Feature logic tests ───────────────────────────────────────────────────────

def test_ever_late_flags_zero():
    result = engineer_features(_sample())
    assert result["ever_late_30"].iloc[0] == 0
    assert result["ever_late_60"].iloc[0] == 0
    assert result["ever_late_90"].iloc[0] == 0


def test_ever_late_flags_nonzero():
    raw = _sample()
    raw["NumberOfTime30-59DaysPastDueNotWorse"] = 2
    result = engineer_features(raw)
    assert result["ever_late_30"].iloc[0] == 1


def test_total_late():
    raw = _sample()
    raw["NumberOfTime30-59DaysPastDueNotWorse"] = 1
    raw["NumberOfTime60-89DaysPastDueNotWorse"] = 2
    raw["NumberOfTimes90DaysLate"]              = 3
    result = engineer_features(raw)
    assert result["total_late"].iloc[0] == 6


def test_late_severity():
    raw = _sample()
    raw["NumberOfTime30-59DaysPastDueNotWorse"] = 2   # ×1 = 2
    raw["NumberOfTime60-89DaysPastDueNotWorse"] = 1   # ×2 = 2
    raw["NumberOfTimes90DaysLate"]              = 3   # ×4 = 12
    result = engineer_features(raw)
    assert result["late_severity"].iloc[0] == 16


def test_log_transforms_nonnegative():
    result = engineer_features(_sample())
    for col in ["MonthlyIncome_log", "DebtRatio_log",
                "RevolvingUtilizationOfUnsecuredLines_log",
                "late_severity_log", "monthly_debt_est_log"]:
        assert result[col].iloc[0] >= 0, f"{col} should be non-negative"


def test_age_group_codes():
    for age, expected in [(25, 0), (35, 1), (50, 2), (65, 3), (80, 4)]:
        raw = _sample()
        raw["age"] = age
        result = engineer_features(raw)
        assert result["age_group_code"].iloc[0] == expected, f"age={age}"


def test_util_group_codes():
    for util, expected in [(0.1, 0), (0.4, 1), (0.7, 2), (0.85, 3), (0.95, 4)]:
        raw = _sample()
        raw["RevolvingUtilizationOfUnsecuredLines"] = util
        result = engineer_features(raw)
        assert result["util_group_code"].iloc[0] == expected, f"util={util}"


def test_no_nans_in_output():
    result = engineer_features(_sample())
    assert not result.isnull().any().any(), "Output should have no NaN values"
