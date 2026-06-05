from fastapi import FastAPI
from fastapi.routing import APIRouter

# 미개발 주석 처리
#from api.v1.predict.predict_router import router as predict_router
#from api.v1.heatmap.heatmap_router import router as heatmap_router
#from api.v1.admin.admin_router import router as admin_router
#from ml.predictor import Predictor

app = FastAPI(title="Flood ML API")

# --- 미완성 부분 더미 라우터로 임시 대체 ---
predict_router = APIRouter()
@predict_router.get("/")
def get_predict():
    return {"message": "predict router stub"}

heatmap_router = APIRouter()
@heatmap_router.get("/")
def get_heatmap():
    return {"message": "heatmap router stub"}

admin_router = APIRouter()
@admin_router.get("/")
def get_admin():
    return {"message": "admin router stub"}
# --------------------------------------------

@app.on_event("startup")
async def load_model():
    #app.state.predictor = Predictor.load_lastest()
    print("[python API] 임시 모델 로드 완료")

app.include_router(predict_router, prefix="/api/v1/predict")
app.include_router(heatmap_router, prefix="/api/v1/heatmap")
app.include_router(admin_router, prefix="/api/v1/admin")