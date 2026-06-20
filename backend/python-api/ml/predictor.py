from pathlib import Path

import lightgbm as lgb

from ml.config import FEATURE_COLS, DEFAULT_THRESHOLD


BASE_DIR = Path(__file__).resolve().parent


class Predictor:
    def __init__(self, model, threshold, feature_cols):
        self.model = model
        self.threshold = threshold
        self.feature_cols = feature_cols

    @classmethod
    def load_latest(cls):
        model_path = BASE_DIR /"model"/ "flood_lgbm_model2.txt"

        model = lgb.Booster(model_file=str(model_path))

        return cls(
            model=model,
            threshold=DEFAULT_THRESHOLD,
            feature_cols=FEATURE_COLS,
        )

    def predict(self, feature_df):
        X = feature_df[self.feature_cols]

        prob = self.model.predict(X)
        pred = (prob >= self.threshold).astype(int)

        return pred, prob