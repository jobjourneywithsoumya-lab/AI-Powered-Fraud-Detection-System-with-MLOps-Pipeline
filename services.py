import os

import numpy as np
import pandas as pd
import joblib


# For local use we load the deployed pipeline directly from the repository
# instead of going through the MLflow model registry (which may reference
# absolute paths from the original author's machine).
_BASE_DIR = os.path.dirname(os.path.dirname(__file__))
_MODEL_PATH = os.path.join(_BASE_DIR, "Notebooks", "fraud_pipeline_deployed.pkl")

if not os.path.exists(_MODEL_PATH):
    raise FileNotFoundError(
        f"Trained model file not found at '{_MODEL_PATH}'. "
        "Make sure you've pulled the full repository (including the Notebooks folder) "
        "or re-run training to generate 'fraud_pipeline_deployed.pkl'."
    )

_model = joblib.load(_MODEL_PATH)


def predict_fraud(data: dict) -> int:
    df = pd.DataFrame([data])
    df = df.where(pd.notnull(df), np.nan)  # converts None to NaN
    preds = _model.predict(df)
    return int(preds[0])
