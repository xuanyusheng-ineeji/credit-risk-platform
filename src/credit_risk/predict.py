"""
Model loading and prediction utilities.
XGBoost predictions use the native Booster API (pred_contribs) to avoid
sklearn-wrapper compatibility issues with older pickled models.
"""
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from typing import Literal

_MODELS_DIR = Path(__file__).parent.parent.parent / "models"

# Module-level singletons (lazy-loaded once on first call)
_lr_model     = None
_xgb_booster  = None
_lr_thresh    = 0.5
_xgb_thresh   = 0.5
_loaded       = False


def load_models(models_dir: Path = _MODELS_DIR) -> None:
    """Load all models into module-level singletons. Safe to call multiple times."""
    global _lr_model, _xgb_booster, _lr_thresh, _xgb_thresh, _loaded
    if _loaded:
        return

    lr_path   = models_dir / "logistic_model.pkl"
    xgb_path  = models_dir / "xgboost_model.pkl"
    lr_thr    = models_dir / "lr_threshold.pkl"
    xgb_thr   = models_dir / "xgb_threshold.pkl"

    if not lr_path.exists() or not xgb_path.exists():
        raise FileNotFoundError(
            f"Model files not found in {models_dir}. "
            "Run the training notebooks first."
        )

    _lr_model = joblib.load(lr_path)

    xgb_raw = joblib.load(xgb_path)
    # Remove deprecated parameter from old XGBoost pickles
    xgb_raw.__dict__.pop("use_label_encoder", None)
    _xgb_booster = xgb_raw.get_booster()

    _lr_thresh  = float(joblib.load(lr_thr))  if lr_thr.exists()  else 0.5
    _xgb_thresh = float(joblib.load(xgb_thr)) if xgb_thr.exists() else 0.5

    _loaded = True


def _risk_level(prob: float) -> str:
    if prob < 0.10:
        return "Low"
    if prob < 0.30:
        return "Medium"
    return "High"


def predict_single(
    features_df: pd.DataFrame,
    model: Literal["xgboost", "lr"] = "xgboost",
) -> dict:
    """
    Predict default probability for a single customer.

    Parameters
    ----------
    features_df : DataFrame with 35 engineered features (1 row).
    model       : 'xgboost' (default) or 'lr'.

    Returns
    -------
    dict with keys:
        probability  – float [0, 1]
        prediction   – int  {0, 1}
        risk_level   – str  {'Low', 'Medium', 'High'}
        threshold    – float
        shap_values  – dict {feature: shap_value}  (XGBoost only)
        bias         – float (SHAP base value / LR intercept)
    """
    load_models()

    if model == "xgboost":
        dmat  = xgb.DMatrix(features_df, feature_names=list(features_df.columns))
        prob  = float(_xgb_booster.predict(dmat)[0])
        thresh = _xgb_thresh
        contribs = _xgb_booster.predict(dmat, pred_contribs=True)[0]
        shap_vals = dict(zip(features_df.columns, contribs[:-1].tolist()))
        bias = float(contribs[-1])
    else:
        prob  = float(_lr_model.predict_proba(features_df)[0, 1])
        thresh = _lr_thresh
        coef  = _lr_model.named_steps["model"].coef_[0]
        shap_vals = dict(zip(features_df.columns, coef.tolist()))
        bias = float(_lr_model.named_steps["model"].intercept_[0])

    return {
        "probability": round(prob, 4),
        "prediction":  int(prob >= thresh),
        "risk_level":  _risk_level(prob),
        "threshold":   thresh,
        "shap_values": shap_vals,
        "bias":        bias,
    }


def predict_batch(
    features_df: pd.DataFrame,
    model: Literal["xgboost", "lr"] = "xgboost",
) -> pd.DataFrame:
    """
    Batch predict for multiple customers.

    Returns the input DataFrame with three new columns:
        default_probability, prediction, risk_level
    """
    load_models()

    if model == "xgboost":
        dmat  = xgb.DMatrix(features_df, feature_names=list(features_df.columns))
        proba = _xgb_booster.predict(dmat)
        thresh = _xgb_thresh
    else:
        proba  = _lr_model.predict_proba(features_df)[:, 1]
        thresh = _lr_thresh

    out = features_df.copy()
    out["default_probability"] = np.round(proba, 4)
    out["prediction"]          = (proba >= thresh).astype(int)
    out["risk_level"]          = pd.cut(
        proba,
        bins=[-0.001, 0.10, 0.30, 1.001],
        labels=["Low", "Medium", "High"],
    )
    return out
