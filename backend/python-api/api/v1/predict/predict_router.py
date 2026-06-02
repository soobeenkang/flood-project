from fastapi import APIRouter, Request
import pandas as pd

router = APIRouter()


@router.post("/")
def predict(request: Request, payload: list[dict]):
    predictor = request.app.state.predictor

    df = pd.DataFrame(payload)

    pred, prob = predictor.predict(df)

    return {
        "prediction": pred.tolist(),
        "probability": prob.tolist()
    }