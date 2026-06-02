import pandas as pd
from ml.predictor import predictor


def predict_flood(input_data: list[dict]):
    df = pd.DataFrame(input_data)

    pred, prob = predictor.predict(df)

    df["flood"] = pred
    df["probability"] = prob

    return df.to_dict(orient="records")