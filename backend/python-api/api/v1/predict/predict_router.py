from fastapi import APIRouter, Request
from pydantic import BaseModel

from api.v1.predict.predict_service import predict_flood


router = APIRouter()


class PredictRequest(BaseModel):
    data: list[dict]


@router.post("/")
def predict(request: Request, body: PredictRequest):
    predictor = request.app.state.predictor

    result = predict_flood(body.data, predictor)

    return {
        "message": "success",
        "data": result
    }