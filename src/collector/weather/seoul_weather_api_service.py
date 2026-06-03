"""

프론트엔드에서 위경도 입력 시

1. 캐시 확인: 이미 조회된 데이터인지, 이미 조회된 이력 있으면 그거 바로 보내기
2. 격자 매핑: 입력받은 위경도에 해당하는 격자(grid_id) 찾기
3. 날씨 쿼리: seoul_weather_collector.py가 생성한 seoul_weather.parquet파일에서 해당 격자에 해당하는 정보 가져오기
4. response 형식 맞춰 데이터 가공 및 보내기

"""

import math
import json
import logging
from typing import Optional
from datetime import datetime
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import redis

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FloodAlert-WeatherAPI")

# FastAPI 인스턴스 생성, app 객체 통해 uvicorn 구동 및 라우팅
app = FastAPI(title="FloodAlert Weather Service", version="1.0")

# Redis 연결 설정
try:
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
except Exception as e:
    logger.warning(f"Redis 연결 실패 (캐시 없이 진행): {e}")
    r = None

PARQUET_PATH = "seoul_weather.parquet"
GEOJSON_PATH = "seoul_grid.geojson"

# ────────────────────────────────────────────────
# 1. Pydantic 모델 정의 (강수확률 제외, grid_id 정수형 반영)
# ────────────────────────────────────────────────
class WeatherResponse(BaseModel):
    grid_id: int        # 그리드 아이디
    rainfall: float     # 현재 강수량 (mm/h)
    temperature: float  # 기온 (°C)
    windSpeed: float    # 풍속 (m/s)
    forecast: str       # 단기 예보 문구
    updatedAt: str      # 데이터 갱신 시각 (ISO 8601)

# ────────────────────────────────────────────────
# 2. 헬퍼 함수: 1시간 후 하늘상태(sky_1h) 기반 예보 문구 생성
# ────────────────────────────────────────────────
def generate_forecast_text(sky_1h: float, rn1_now: float) -> str:
    """
    1시간 후 하늘상태(sky_1h) 코드값을 해석하여 단기 예보 문구를 생성
    (1=맑음, 2=구름조금, 3=구름많음, 4=흐림)
    """
    sky_status = {
        1.0: "맑음",
        2.0: "구름조금",
        3.0: "구름많음",
        4.0: "흐림"
    }.get(sky_1h, "흐림")

    if rn1_now > 0:
        return f"현재 시간당 {rn1_now:.1f}mm의 비가 내리고 있으며, 1시간 후 기상 상태는 '{sky_status}' 상태로 예측됩니다."
    
    return f"오후 기상 예보: 1시간 뒤 하늘 상태는 '{sky_status}' 상태로 예측됩니다."

# ────────────────────────────────────────────────
# 3. API 엔드포인트 구현: GET /v1/weather/current
# ────────────────────────────────────────────────
@app.get("/v1/weather/current", response_model=WeatherResponse)
def get_current_weather(
    lat: float = Query(..., description="위도 (WGS84)", example=37.5571),
    lon: float = Query(..., description="경도 (WGS84)", example=126.9368)
):
    # 0. 에러 핸들링: 위경도 유효성 검사
    if not (33.0 <= lat <= 43.0 and 124.0 <= lon <= 132.0):
        raise HTTPException(
            status_code=400, 
            detail={"code": "INVALID_COORDINATES", "message": "위도·경도 범위가 비정상적입니다."}
        )

    # 1. Redis 캐시 확인 (TTL 10분)
    cache_key = f"weather:{round(lat, 4)}:{round(lon, 4)}"
    if r: # redis 서버가 정상적으로 켜져서 연결된 상태일 때
        try:
            cached_data = r.get(cache_key)
            if cached_data:
                logger.info("Redis 캐시 히트")
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Redis 읽기 실패: {e}")

    # 변수 안전성 확보를 위한 초기화
    target_grid_id = None

    # 2. GeoJSON 기반 최접근 격자(grid_id) 탐색
    try:
        with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
            gj = json.load(f)
            
        # 요청된 위경도와 가장 거리가 가까운 격자 Feature 선택
        closest_feat = min(
            gj["features"], 
            key=lambda x: math.hypot(x["properties"]["lat"] - lat, x["properties"]["lon"] - lon)
        )
        
        # 소수점 형태나 문자열 데이터가 들어와도 안전하게 정수로 변환되도록
        raw_grid_id = closest_feat["properties"]["grid_id"]
        target_grid_id = int(float(raw_grid_id))
        
    except Exception as e:
        logger.error(f"GeoJSON 매핑 혹은 ID 파싱 실패: {e}")
        raise HTTPException(
            status_code=404, 
            detail={"code": "GRID_NOT_FOUND", "message": f"그리드 파싱 실패: {str(e)}"}
        )

    # 3. seoul_weather.parquet 파일 조회 및 데이터 필터링
    try:
        df = pd.read_parquet(PARQUET_PATH)
        
        # parquet 데이터 내 grid_id 컬럼과 정확히 비교하기 위해 int 형식 유지
        grid_data = df[df["grid_id"] == target_grid_id]
        
        if grid_data.empty:
            raise HTTPException(
                status_code=404, 
                detail={"code": "GRID_NOT_FOUND", "message": f"격자 ID {target_grid_id}에 해당하는 날씨 데이터가 데이터셋에 없습니다."}
            )
        
        row = grid_data.iloc[0]
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Parquet 로드 혹은 쿼리 에러: {e}")
        raise HTTPException(
            status_code=500, 
            detail={"code": "INTERNAL_ERROR", "message": f"서버 내부 데이터 처리 실패: {str(e)}"}
        )

    # 4. 수집 데이터 바인딩 및 타입 안전성 확보 (소수점 첫째 자리 반올림)
    rn1_now = round(float(row["rn1_now"]), 1) if pd.notna(row["rn1_now"]) else 0.0
    t1h_now = round(float(row["t1h_now"]), 1) if pd.notna(row["t1h_now"]) else 0.0
    wsd_now = round(float(row["wsd_now"]), 1) if pd.notna(row["wsd_now"]) else 0.0
    sky_1h  = float(row["sky_1h"]) if pd.notna(row["sky_1h"]) else 4.0

    # 5. 동적 단기 예보 문구(forecast) 확정
    forecast_text = generate_forecast_text(sky_1h, rn1_now)

    # 6. 응답 페이로드 조립
    response_payload = {
        "grid_id": target_grid_id,
        "rainfall": rn1_now,         
        "temperature": t1h_now,      
        "windSpeed": wsd_now,        
        "forecast": forecast_text,   
        "updatedAt": pd.to_datetime(row["tmfc"]).isoformat() + "+09:00"
    }

    # 7. Redis 캐싱 저장 (TTL 10분 = 600초)
    if r:
        try:
            r.setex(cache_key, 600, json.dumps(response_payload))
        except Exception as e:
            logger.warning(f"Redis 캐시 저장 실패: {e}")

    return response_payload