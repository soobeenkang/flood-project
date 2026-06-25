import asyncio
from fastapi import FastAPI
from fastapi.routing import APIRouter
from apscheduler.schedulers.background import BackgroundScheduler

# 미개발 주석 처리
#from api.v1.predict.predict_router import router as predict_router
#from api.v1.heatmap.heatmap_router import router as heatmap_router
#from api.v1.admin.admin_router import router as admin_router
#from ml.predictor import Predictor

# 💡 이 줄이 누락되었거나 주석(#) 처리되어 있는지 확인하고 넣어주세요!
from ml.collector.weather.seoul_weather_collector import collect_weather_data

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

scheduler = BackgroundScheduler(timezone="Asia/Seoul")

@app.on_event("startup")
async def startup_event():
    # 1. 기존 임시 모델 로드 로직
    # app.state.predictor = Predictor.load_lastest()
    print("[python API] 임시 모델 로드 완료")

    # 2. ⏰ 날씨 수집 스케줄러 시작
    # 'cron' 모드로 minute=0을 주면 매 시 정각(0분)마다 실행됩니다.
    scheduler.add_job(collect_weather_data, 'cron', minute=0, id='kma_weather_job')
    scheduler.start()
    print("[python API] 기상청 날씨 수집 스케줄러 가동 시작 (매 정각)")

    print("[python API] 서버 시작 기동: 초기 날씨 데이터 수집을 즉시 시작")

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, collect_weather_data)


@app.on_event("shutdown")
async def shutdown_event():
    # 서버가 꺼질 때 스케줄러도 안전하게 종료해줍니다.
    if scheduler.running:
        scheduler.shutdown()
        print("[python API] 날씨 수집 스케줄러가 안전하게 종료됨")

'''
@app.on_event("startup")
async def load_model():
    #app.state.predictor = Predictor.load_lastest()
    print("[python API] 임시 모델 로드 완료")
'''

app.include_router(predict_router, prefix="/api/v1/predict")
app.include_router(heatmap_router, prefix="/api/v1/heatmap")
app.include_router(admin_router, prefix="/api/v1/admin")