from fastapi import FastAPI
from api.v1.predict.predict_router import router as predict_router
from api.v1.heatmap.heatmap_router import router as heatmap_router
from api.v1.admin.admin_router import router as admin_router
from ml.predictor import Predictor

app = FasAPI(title="Flood ML API")

@app.on_event("startup")
async def load_model():
    app.state.predictor = Predictor.load_lastest()

app.include_router(predict_router, prefix="/api/v1/predict")
app.include_router(heatmap_router, prefix="/api/v1/heatmap")
app.include_router(admin_router, prefix="/api/v1/admin")