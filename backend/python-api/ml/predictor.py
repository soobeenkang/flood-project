import joblib
from pathlib import Path


class Predictor:
    def __init__(self, model):
        self.model = model

    @classmethod
    def load_latest(cls):
        model_path = Path("ml/model/flood_model.pkl")
        model = joblib.load(model_path)
        return cls(model)

    def predict(self, df):
        prob = self.model.predict_proba(df)[:, 1]
        pred = (prob >= 0.5).astype(int)
        return pred, prob