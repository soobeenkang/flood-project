import json
from pathlib import Path

import joblib


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"


class Predictor:
    def __init__(self, model, threshold, feature_cols):
        self.model = model
        self.threshold = threshold
        self.feature_cols = feature_cols

    @classmethod
    def load_latest(cls):
        model_path = MODEL_DIR / "xgboost_flood.pkl"
        meta_path = MODEL_DIR / "xgboost_flood_meta.json"
        feature_cols_path = MODEL_DIR / "feature_cols.json"

        model = joblib.load(model_path)

        with open(feature_cols_path, "r", encoding="utf-8") as f:
            feature_cols = json.load(f)

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        feature_cols = meta["feature_cols"]
        threshold = meta.get("best_threshold", 0.5)

        return cls(model, threshold, feature_cols)

    def predict(self, feature_df):
        X = feature_df[self.feature_cols]

        prob = self.model.predict_proba(X)[:, 1]
        pred = (prob >= self.threshold).astype(int)

        return pred, prob