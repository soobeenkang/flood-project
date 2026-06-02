from fastapi import FastAPI
from fastapi.routing import APIRouter

# 실제 구현된 Predict API 라우터
from api.v1.predict.predict_router import router as predict_router

# 모델 로더
from ml.predictor import Predictor

app = FastAPI(title="Flood ML API")


# =========================
# 미구현 API 임시 Stub
# heatmap / admin 기능 개발 전까지 사용
# =========================

heatmap_router = APIRouter()

@heatmap_router.get("/")
def get_heatmap():
    return {"message": "heatmap router stub"}


admin_router = APIRouter()

@admin_router.get("/")
def get_admin():
    return {"message": "admin router stub"}


# =========================
# 서버 시작 시 모델 로드
# =========================
@app.on_event("startup")
async def load_model():
    app.state.predictor = Predictor.load_latest()
    print("[python API] 모델 로드 완료")


# =========================
# API Router 등록
# =========================
app.include_router(predict_router, prefix="/api/v1/predict")
app.include_router(heatmap_router, prefix="/api/v1/heatmap")
app.include_router(admin_router, prefix="/api/v1/admin")