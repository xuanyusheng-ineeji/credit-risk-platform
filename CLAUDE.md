# CLAUDE.md — Credit Risk Assessment Platform

## Environment

Conda environment name: `credit-risk` (Python 3.10).

All commands must be run with conda activated. On Windows, multiline `-c` scripts fail with conda; write to a temp file first:

```powershell
# Wrong — conda rejects multiline -c
conda run -n credit-risk python -c "
import pandas
..."

# Correct — write to file, then run
# Write script to scratchpad, then:
conda run -n credit-risk python path/to/script.py
```

Editable install is required for `import credit_risk` to work:

```bash
pip install -e .
```

---

## Common Commands

### Run the Streamlit app

```bash
conda run -n credit-risk streamlit run app/Home.py
# http://localhost:8501
```

### Run tests

```bash
# Feature engineering unit tests (no model files required — fast, safe for CI)
conda run -n credit-risk pytest tests/test_features.py -v

# All tests (prediction tests auto-skip if models/ absent)
conda run -n credit-risk pytest tests/ -v

# Coverage (must stay >= 70%)
conda run -n credit-risk pytest tests/test_features.py --cov=credit_risk --cov-report=term-missing
```

### Lint

```bash
conda run -n credit-risk flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503
```

### Docker

```bash
docker compose up --build -d      # Build and start
docker compose logs -f            # Stream logs
docker compose down               # Stop
```

### Train models (if models/ is empty)

Run notebooks in order:
1. `notebooks/02_data_cleaning.ipynb`
2. `notebooks/03_feature_engineering.ipynb`
3. `notebooks/04_logistic_regression.ipynb`
4. `notebooks/05_xgboost_model.ipynb`

---

## Architecture

```
src/credit_risk/     ← installable Python package (the core library)
  features.py        ← pure feature engineering, no model dependency
  predict.py         ← model loading + inference, depends on features.py
  __init__.py        ← exports: engineer_features, FEATURE_ORDER, RAW_FEATURES,
                               load_models, predict_single, predict_batch

app/                 ← Streamlit multipage app, imports from credit_risk
  Home.py            ← landing page
  pages/
    1_Single_Prediction.py
    2_Batch_Prediction.py
    3_Dashboard.py

notebooks/           ← research and training only, not imported by app
models/              ← .pkl files, NOT in Docker image (volume-mounted)
data/processed/      ← pre-processed CSVs, NOT in Docker image (volume-mounted)
tests/               ← pytest; test_features.py needs no models
```

Each `app/pages/*.py` file adds `src/` to `sys.path` with 3 levels up (`__file__` → pages → app → project root → src). `Home.py` uses 2 levels up.

---

## Feature Engineering Rules

`FEATURE_ORDER` in `src/credit_risk/features.py` is the **exact 35-feature column order** the models were trained on. Any deviation causes silent wrong predictions.

- Never reorder `FEATURE_ORDER`.
- `engineer_features()` always returns `df[FEATURE_ORDER]` — this is the guarantee.
- When adding a new engineered feature, it must be appended to the END of `FEATURE_ORDER` and models must be retrained.

The 10 raw input feature names are in `RAW_FEATURES`. Both lists are exported from `__init__.py`.

---

## XGBoost Compatibility Rules

**Always use the native Booster API for inference, never `XGBClassifier.predict_proba()`.**

```python
# Loading (in predict.py load_models())
xgb_raw = joblib.load(xgb_path)
xgb_raw.__dict__.pop("use_label_encoder", None)   # must come first
booster = xgb_raw.get_booster()

# Inference
dmat = xgb.DMatrix(features_df, feature_names=list(features_df.columns))
proba = booster.predict(dmat)                      # probabilities
contribs = booster.predict(dmat, pred_contribs=True)  # TreeSHAP
shap_values = contribs[:, :-1]                     # last column is bias
```

**Why:** XGBoost 2.x replaced `use_label_encoder` with a `@property`; old pickled models have it in `__dict__`, causing `AttributeError` when `get_params()` iterates it. Never call sklearn wrapper methods on loaded models.

**SHAP:** Do NOT use `shap.TreeExplainer`. XGBoost 2.x stores `base_score` internally as `'[5E-1]'` (with brackets); the SHAP library's `float(...)` call fails on it. Use `pred_contribs=True` exclusively.

---

## Prediction API

```python
from credit_risk import engineer_features, predict_single, predict_batch

# Single customer (dict → 35 features → prediction)
features_df = engineer_features(raw_dict)
result = predict_single(features_df, model="xgboost")
# result keys: probability, prediction, risk_level, threshold, shap_values (dict, 35 entries), bias

# Batch (DataFrame → 35 features → predictions)
features_df = engineer_features(df_raw)
out_df = predict_batch(features_df, model="xgboost")
# adds columns: default_probability, prediction, risk_level
```

Risk levels: `Low` < 0.10, `Medium` 0.10–0.30, `High` ≥ 0.30.

`load_models()` is a lazy singleton — safe to call multiple times. Raises `FileNotFoundError` if `models/xgboost_model.pkl` or `models/logistic_model.pkl` is missing.

---

## Testing Conventions

- `tests/test_features.py` — 21 unit tests, zero model dependency. Always run these first.
- `tests/test_predict.py` — 9 integration tests. They are automatically skipped (via `conftest.py` `MODELS_PRESENT` flag) if `models/xgboost_model.pkl` does not exist. Do not mark them as failures if models are absent.
- Coverage target: ≥ 70% (enforced in CI with `--cov-fail-under=70`).
- `conftest.py` only detects XGBoost model presence; LR model absence does not trigger a skip.

---

## Notebook Conventions

- Edit `.ipynb` files with the `NotebookEdit` tool (not the `Edit` tool).
- Cell IDs are used by `NotebookEdit` to target specific cells.
- Notebooks are research artifacts; production logic lives in `src/credit_risk/`.

---

## Model Files

| File | Size | Contents |
|------|------|----------|
| `models/logistic_model.pkl` | ~3.5 KB | sklearn Pipeline (StandardScaler → LogisticRegression) |
| `models/xgboost_model.pkl` | ~283 KB | XGBClassifier (extract `.get_booster()` before use) |
| `models/xgb_threshold.pkl` | ~117 B | float decision threshold |
| `models/lr_threshold.pkl` | absent | defaults to 0.5 in `load_models()` |

---

## Data Files

| File | Rows | Columns | Notes |
|------|------|---------|-------|
| `data/processed/train_clean.csv` | 149,391 | 11 | Cleaned raw features + target |
| `data/processed/train_features.csv` | 149,391 | 36 | 35 engineered features + target |
| `data/processed/test_clean.csv` | 101,503 | 10 | No target column |

---

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`):
- Matrix: Python 3.10, 3.11
- Steps: install → flake8 → `test_features` → `test_predict` → coverage → codecov (3.10 only)
- Second job: Docker build (no push), depends on test job passing

---

## Key Metrics (reports/model_comparison.csv)

| Model | ROC-AUC | PR-AUC | Gini | KS | F1 | Recall | Precision |
|-------|---------|--------|------|----|----|--------|-----------|
| Logistic Regression | 0.8597 | 0.3813 | 0.7193 | 0.5735 | 0.3338 | 0.7807 | 0.2122 |
| XGBoost (tuned) | 0.8659 | 0.4045 | 0.7318 | 0.585 | 0.4591 | 0.52 | 0.411 |
