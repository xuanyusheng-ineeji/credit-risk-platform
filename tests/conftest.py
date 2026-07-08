from pathlib import Path

MODELS_PRESENT = (Path(__file__).parent.parent / "models" / "xgboost_model.pkl").exists()
