import requests
import csv
import os
from datetime import datetime, timedelta
import math
import time


# 위경도 → 격자 변환
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


# 강남구 격자 좌표
NX_TOTAL    = 149
NY_TOTAL    = 253
GANGNAM_LAT = 37.5172
GANGNAM_LON = 127.0473

# 격자 좌표 → 응답 배열 인덱스 변환
NX, NY      = latlon_to_grid(GANGNAM_LAT, GANGNAM_LON)
TARGET_IDX  = (NY - 1) * NX_TOTAL + (NX - 1)

# 그리드 ID: "NX{nx}_NY{ny}" 형식
GRID_ID     = f"NX{NX}_NY{NY}"


# 발표시간 계산
# 초단기예보: 매 10분 발표, 발효시간은 +1h~+6h, 최소 20분 이전 발표시간 사용
def get_forecast_times(base_dt: datetime):
    adjusted  = base_dt - timedelta(minutes=20)
    minute    = (adjusted.minute // 10) * 10
    tmfc_dt   = adjusted.replace(minute=minute, second=0, microsecond=0)
    tmfc_str  = tmfc_dt.strftime("%Y%m%d%H%M")

    # 발효시간: +1h ~ +6h
    tmef_list = []
    for i in range(1, 7):
        tmef_dt = tmfc_dt + timedelta(hours=i)
        tmef_list.append((tmef_dt, tmef_dt.strftime("%Y%m%d%H")))

    return tmfc_str, tmfc_dt, tmef_list


# API 응답 파싱
def parse_grid_response(text: str) -> list:
    values = []
    for token in text.replace('\n', ',').split(','):
        t = token.strip()
        if t:
            try:
                values.append(float(t))
            except ValueError:
                pass
    return values


BASE_URL = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-dfs_vsrt_grd"
AUTH_KEY  = ""


# 단일 변수 API 호출 → 강남구 해당 인덱스 값 반환
def fetch_value(tmfc: str, tmef: str, var: str) -> float | None:
    params = {
            "tmfc":     tmfc, 
            "tmef":     tmef, 
            "vars":     var, 
            "authKey":  AUTH_KEY,
    }
    try:
        resp = requests.get(BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        values = parse_grid_response(resp.text)

        total = NX_TOTAL * NY_TOTAL  # 37,697
        if len(values) < total:
            print(f"  [경고] {var} {tmef}: 값 개수 부족 ({len(values)}/{total})")
            if TARGET_IDX < len(values):
                val = values[TARGET_IDX]
                return None if val <= -99.0 else val
            return None

        val = values[TARGET_IDX]
        return None if val <= -99.0 else val

    except requests.RequestException as e:
        print(f"  [오류] {var} {tmef}: {e}")
        return None


# 누적 계산용
def safe_sum(vals: list) -> float | None:
    filtered = [v for v in vals if v is not None]
    return round(sum(filtered), 1) if filtered else None


def main():
    now = datetime.now()
    tmfc, tmfc_dt, tmef_list = get_forecast_times(now)

    # 강수데이터 수집 (+1h ~ +6h)
    rn1_values = []
    for _, tmef in tmef_list:
        rn1_values.append(fetch_value(tmfc, tmef, "RN1"))
        time.sleep(0.3)

    # 현재 강수량 = +1h RN1 (직전 1시간)
    current_rain = rn1_values[0]

    # 누적 강수량
    acc_1h = safe_sum(rn1_values[:1])
    acc_3h = safe_sum(rn1_values[:3])
    acc_6h = safe_sum(rn1_values[:6])

    # CSV 저장 (같은 디렉터리에 gangnam_rain.csv, 실행마다 행 추가)
    output_dir  = os.path.dirname(os.path.abspath(__file__))
    csv_path    = os.path.join(output_dir, "gangnam_rain.csv")
    file_exists = os.path.isfile(csv_path)

    fieldnames = [
        "grid_id",
        "timestamp",
        "current_rain_mm",
        "acc_1h_mm",
        "acc_3h_mm",
        "acc_6h_mm",
        "future_rain_1h",
        "future_rain_2h",
        "future_rain_3h",
        "future_rain_4h",
        "future_rain_5h",
        "future_rain_6h",
    ]

    row = {
        "grid_id":         GRID_ID,
        "timestamp":       now.strftime("%Y-%m-%d %H:%M"),
        "current_rain_mm": current_rain,
        "acc_1h_mm":       acc_1h,
        "acc_3h_mm":       acc_3h,
        "acc_6h_mm":       acc_6h,
        "future_rain_1h":  rn1_values[0],
        "future_rain_2h":  rn1_values[1],
        "future_rain_3h":  rn1_values[2],
        "future_rain_4h":  rn1_values[3],
        "future_rain_5h":  rn1_values[4],
        "future_rain_6h":  rn1_values[5],
    }

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
