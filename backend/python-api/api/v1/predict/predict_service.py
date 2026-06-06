from ml.preprocessing.build_feature_api import build_features_for_api


def predict_flood(raw_data, predictor):
    feature_df = build_features_for_api(raw_data)

    pred, prob = predictor.predict(feature_df)

    return [
        {
            "prediction": int(p),
            "probability": float(pr)
        }
        for p, pr in zip(pred, prob)
    ]