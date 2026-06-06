from fastapi import FastAPI
from fastapi.routing import APIRouter

from api.v1.predict.predict_router import router as predict_router
from ml.predictor import Predictor

app = FastAPI(title="Flood ML API")


# 아직 구현 안 된 API만 stub 유지
heatmap_router = APIRouter()

@heatmap_router.get("/")
def get_heatmap():
    return {"message": "heatmap router stub"}


admin_router = APIRouter()

@admin_router.get("/")
def get_admin():
    return {"message": "admin router stub"}


@app.on_event("startup")
async def load_model():
    app.state.predictor = Predictor.load_latest()
    print("[python API] 모델 로드 완료")


app.include_router(predict_router, prefix="/api/v1/predict")
app.include_router(heatmap_router, prefix="/api/v1/heatmap")
app.include_router(admin_router, prefix="/api/v1/admin")