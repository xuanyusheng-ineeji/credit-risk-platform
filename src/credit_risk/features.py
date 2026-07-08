"""
Feature engineering pipeline — reproduces the exact same transformations
applied in notebooks 02_data_cleaning and 03_feature_engineering.
"""
import numpy as np
import pandas as pd
from typing import Union

# ── Preprocessing constants (computed from training data at 99.9th percentile) ──
DEBT_RATIO_CAP  = 10640.54
INCOME_CAP      = 72817.42
AGE_MEDIAN      = 52.0
INCOME_MEDIAN   = 5400.0
DEP_MEDIAN      = 0.0

RAW_FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

LATE_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]

# Exact feature order the model was trained on
FEATURE_ORDER = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
    "MonthlyIncome_missing",
    "NumberOfDependents_missing",
    "age_missing",
    "NumberOfTime30-59DaysPastDueNotWorse_missing",
    "NumberOfTime60-89DaysPastDueNotWorse_missing",
    "NumberOfTimes90DaysLate_missing",
    "ever_late_30",
    "ever_late_60",
    "ever_late_90",
    "total_late",
    "MonthlyIncome_log",
    "DebtRatio_log",
    "RevolvingUtilizationOfUnsecuredLines_log",
    "age_group_code",
    "util_group_code",
    "monthly_debt_est",
    "monthly_debt_est_log",
    "debt_per_credit_line",
    "income_per_dependent",
    "income_per_dependent_log",
    "credit_line_per_age",
    "real_estate_ratio",
    "util_x_late",
    "late_severity",
    "late_severity_log",
]


def engineer_features(raw: Union[dict, pd.DataFrame]) -> pd.DataFrame:
    """
    Transform raw input (10 features) into the 35-feature vector the model expects.

    Parameters
    ----------
    raw : dict or DataFrame
        Must contain the 10 raw feature columns.

    Returns
    -------
    DataFrame with exactly 35 columns in FEATURE_ORDER.
    """
    df = pd.DataFrame([raw]) if isinstance(raw, dict) else raw.copy()

    # ── 1. Missing-value flags ──────────────────────────────────────────────
    df["MonthlyIncome_missing"]       = df["MonthlyIncome"].isna().astype(int)
    df["NumberOfDependents_missing"]  = df["NumberOfDependents"].isna().astype(int)
    df["age_missing"]                 = (df["age"] == 0).astype(int)

    for col in LATE_COLS:
        df[f"{col}_missing"] = df[col].isin([96, 98]).astype(int)
        df.loc[df[col].isin([96, 98]), col] = np.nan

    # ── 2. Fix anomalies ────────────────────────────────────────────────────
    df.loc[df["age"] == 0, "age"] = np.nan

    # ── 3. Clip outliers ────────────────────────────────────────────────────
    df["RevolvingUtilizationOfUnsecuredLines"] = \
        df["RevolvingUtilizationOfUnsecuredLines"].clip(0, 1)
    df["DebtRatio"]      = df["DebtRatio"].clip(upper=DEBT_RATIO_CAP)
    df["MonthlyIncome"]  = df["MonthlyIncome"].clip(upper=INCOME_CAP)

    # ── 4. Fill missing values ───────────────────────────────────────────────
    df["age"]                = df["age"].fillna(AGE_MEDIAN).astype(float)
    df["MonthlyIncome"]      = df["MonthlyIncome"].fillna(INCOME_MEDIAN).astype(float)
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(DEP_MEDIAN).astype(float)
    for col in LATE_COLS:
        df[col] = df[col].fillna(0.0).astype(float)

    # ── 5. Delinquency binary features ──────────────────────────────────────
    df["ever_late_30"] = (df["NumberOfTime30-59DaysPastDueNotWorse"] > 0).astype(int)
    df["ever_late_60"] = (df["NumberOfTime60-89DaysPastDueNotWorse"] > 0).astype(int)
    df["ever_late_90"] = (df["NumberOfTimes90DaysLate"] > 0).astype(int)
    df["total_late"]   = (
        df["NumberOfTime30-59DaysPastDueNotWorse"]
        + df["NumberOfTime60-89DaysPastDueNotWorse"]
        + df["NumberOfTimes90DaysLate"]
    )

    # ── 6. Log transforms ───────────────────────────────────────────────────
    df["MonthlyIncome_log"]                      = np.log1p(df["MonthlyIncome"])
    df["DebtRatio_log"]                          = np.log1p(df["DebtRatio"])
    df["RevolvingUtilizationOfUnsecuredLines_log"] = \
        np.log1p(df["RevolvingUtilizationOfUnsecuredLines"])

    # ── 7. Age group code ───────────────────────────────────────────────────
    age_bins   = [0, 30, 45, 60, 75, 200]
    age_labels = [0, 1, 2, 3, 4]
    df["age_group_code"] = pd.cut(
        df["age"], bins=age_bins, labels=age_labels, right=True
    ).astype(int)

    # ── 8. Utilisation group code ───────────────────────────────────────────
    util_bins   = [-0.001, 0.3, 0.6, 0.8, 0.9, 10.0]
    util_labels = [0, 1, 2, 3, 4]
    df["util_group_code"] = pd.cut(
        df["RevolvingUtilizationOfUnsecuredLines"],
        bins=util_bins, labels=util_labels, right=True
    ).astype(int)

    # ── 9. Ratio / interaction features ────────────────────────────────────
    df["monthly_debt_est"]       = df["DebtRatio"] * df["MonthlyIncome"]
    df["monthly_debt_est_log"]   = np.log1p(df["monthly_debt_est"])
    df["debt_per_credit_line"]   = df["DebtRatio"] / (df["NumberOfOpenCreditLinesAndLoans"] + 1)
    df["income_per_dependent"]   = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)
    df["income_per_dependent_log"] = np.log1p(df["income_per_dependent"])
    df["credit_line_per_age"]    = df["NumberOfOpenCreditLinesAndLoans"] / (df["age"] + 1)
    df["real_estate_ratio"]      = (
        df["NumberRealEstateLoansOrLines"] / (df["NumberOfOpenCreditLinesAndLoans"] + 1)
    )
    df["util_x_late"] = df["RevolvingUtilizationOfUnsecuredLines"] * df["total_late"]

    # ── 10. Late severity ───────────────────────────────────────────────────
    df["late_severity"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"] * 1
        + df["NumberOfTime60-89DaysPastDueNotWorse"] * 2
        + df["NumberOfTimes90DaysLate"] * 4
    )
    df["late_severity_log"] = np.log1p(df["late_severity"])

    return df[FEATURE_ORDER]
