import requests
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
import math
import time

import pandas as pd


# 위경도 → 기상청 격자 변환
def latlon_to_grid(lat, lon):
    RE = 6371.00877
    GRID = 5.0
    SLAT1 = 30.0
    SLAT2 = 60.0
    OLON  = 126.0   
    OLAT  = 38.0
    XO = 43
    YO = 136
    
    DEGRAD = math.pi / 180.0
    re    = RE / GRID
    slat1 = SLAT1 * DEGRAD
    slat2 = SLAT2 * DEGRAD
    olon  = OLON  * DEGRAD
    olat  = OLAT  * DEGRAD

    sn = math.log(math.cos(slat1) / math.cos(slat2)) / \
         math.log(math.tan(math.pi * 0.25 + slat2 * 0.5) /
                  math.tan(math.pi * 0.25 + slat1 * 0.5))
    sf = (math.tan(math.pi * 0.25 + slat1 * 0.5) ** sn) * math.cos(slat1) / sn
    ro = re * sf / (math.tan(math.pi * 0.25 + olat * 0.5) ** sn)
    ra = re * sf / (math.tan(math.pi * 0.25 + lat * DEGRAD * 0.5) ** sn)
    theta = lon * DEGRAD - olon
    if theta >  math.pi: theta -= 2.0 * math.pi
    if theta < -math.pi: theta += 2.0 * math.pi
    theta *= sn

    x = int(ra * math.sin(theta) + XO + 0.5)
    y = int(ro - ra * math.cos(theta) + YO + 0.5)
    return x, y


# 강남구 각 그리드의 위경도를 기상청 격자로 변환해서 어떤 격자에 속하는지 그룹화
# 반환값: { (NX, NY): [그리드 아이디, ...], ... }
def build_kma_to_grid_map(geojson_path: str) -> dict:
    with open(geojson_path, encoding="utf-8") as f:
        gj = json.load(f)

    kma_map = defaultdict(list)
    for feat in gj["features"]:
        p = feat["properties"]
        nx, ny = latlon_to_grid(p["lat"], p["lon"]) #위도 경도를 기상청격자좌표로
        kma_map[(nx, ny)].append(p["grid_id"])

    return dict(kma_map)


NX_TOTAL = 149
NY_TOTAL = 253

# geojason 파일 경로(일단은 현재파일위치로)
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
GEOJSON_PATH = os.path.join(SCRIPT_DIR, "gangnam_grid.geojson")

# 강남구에 걸치는 기상청 격자 5개
KMA_TO_GRID_MAP = build_kma_to_grid_map(GEOJSON_PATH)


# 발표시간 계산
# 초단기예보: 매 10분 발표, 발효시간은 +1h~+6h, 최소 20분 이전 발표시간 사용
def get_forecast_times(base_dt: datetime):
    adjusted = base_dt - timedelta(minutes=20)
    minute   = (adjusted.minute // 10) * 10
    tmfc_dt  = adjusted.replace(minute=minute, second=0, microsecond=0)
    tmfc_str = tmfc_dt.strftime("%Y%m%d%H%M")

    # 발효시간: +1h ~ +6h
    tmef_list = []
    for i in range(1, 7):
        tmef_dt = tmfc_dt + timedelta(hours=i)
        tmef_list.append((tmef_dt, tmef_dt.strftime("%Y%m%d%H")))

    return tmfc_str, tmfc_dt, tmef_list


# API 응답 파싱
def parse_grid_response(text: str) -> list:
    values = []
    for token in text.replace("\n", ",").split(","):
        t = token.strip()
        if t:
            try:
                values.append(float(t))
            except ValueError:
                pass
    return values


BASE_URL = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-dfs_vsrt_grd"
AUTH_KEY  = ""

TIMEOUT = 60    # 초
MAX_RETRIES = 3 # 최대 재시도 횟수
RETRY_WAIT = 5  # 재시도 대기 시간(초)


# 특정 격자 값 가져오기
def fetch_value(tmfc: str, tmef: str, var: str, nx: int, ny: int) -> float | None:
    target_idx = (ny - 1) * NX_TOTAL + (nx - 1)
    params = {
        "tmfc":    tmfc,
        "tmef":    tmef,
        "vars":    var,
        "authKey": AUTH_KEY,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        
        try:
            resp = requests.get(BASE_URL, params=params, timeout=15)
            resp.raise_for_status()
            values = parse_grid_response(resp.text)

            total = NX_TOTAL * NY_TOTAL  # 37,697
            if len(values) < total:
                print(f"  [경고] NX={nx} NY={ny} {var} {tmef}: 값 개수 부족 ({len(values)}/{total})")
                if target_idx < len(values):
                    val = values[target_idx] 
                    return None if val <= -99.0 else val
                return None

            val = values[target_idx]
            return None if val <= -99.0 else val

        except requests.RequestException as e:
            print(f"  [오류] NX={nx} NY={ny} {var} {tmef} (시도 {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT)

    print(f"    [실패] NX={nx} NY={ny} {var} {tmef}: {MAX_RETRIES}회 재시도 후 포기")
    return None


# 누적 계산용
def safe_sum(vals: list) -> float | None:
    filtered = [v for v in vals if v is not None]
    return round(sum(filtered), 1) if filtered else None


def main():
    now = datetime.now()
    tmfc, tmfc_dt, tmef_list = get_forecast_times(now)
    timestamp = now.strftime("%Y-%m-%d %H:%M")

    # 기상청 격자별 강수 데이터 수집
    kma_rain = {}
    for (nx, ny) in KMA_TO_GRID_MAP:
        rn1_values = []
        for _, tmef in tmef_list:
            rn1_values.append(fetch_value(tmfc, tmef, "RN1", nx, ny))
            time.sleep(0.3)
        kma_rain[(nx, ny)] = rn1_values

    # 행 리스트로 각 기상청 격자에 속하는 그리드마다 해당 격자의 강수량 기록
    rows = []
    for (nx, ny), grid_ids in KMA_TO_GRID_MAP.items():
        rn1_values = kma_rain[(nx,ny)]
        current_rain = rn1_values[0]
        acc_1h = safe_sum(rn1_values[:1])
        acc_3h = safe_sum(rn1_values[:3])
        acc_6h = safe_sum(rn1_values[:6])

        for grid_id in grid_ids:
            rows.append({
                "grid_id":          grid_id,
                "timestamp":        timestamp,
                "current_rain_mm":  current_rain,
                "acc_1h_mm":        acc_1h,
                "acc_3h_mm":        acc_3h,
                "acc_6h_mm":        acc_6h,
                "future_rain_1h":   rn1_values[0],
                "future_rain_2h":   rn1_values[1],
                "future_rain_3h":   rn1_values[2],
                "future_rain_4h":   rn1_values[3],
                "future_rain_5h":   rn1_values[4],
                "future_rain_6h":   rn1_values[5],
            })

    # 실행데이터 데이터프레임 생성
    df_new = pd.DataFrame(rows).astype({
        "grid_id":          "int32",
        "current_rain_mm":  "float32",
        "acc_1h_mm":        "float32",
        "acc_3h_mm":        "float32",
        "acc_6h_mm":        "float32",
        "future_rain_1h":   "float32",
        "future_rain_2h":   "float32",
        "future_rain_3h":   "float32",
        "future_rain_4h":   "float32",
        "future_rain_5h":   "float32",
        "future_rain_6h":   "float32",
    })
    df_new["timestamp"] = pd.to_datetime(df_new["timestamp"])

    # 파켓 저장, 기존 파일 있으면 읽고 합치고 다시저장
    parquet_path = os.path.join(SCRIPT_DIR, "gangnam_rain.parquet")
    if os.path.isfile(parquet_path):
        df_existing = pd.read_parquet(parquet_path)
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_parquet(parquet_path, index=False)
    print(f"저장 완료: {parquet_path} ({len(df_all):,}행)")


if __name__ == "__main__":
    main()

