import time
import requests
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

seen = set()

BASE_DIR = Path(__file__).resolve().parents[3]
GRID_MAP_PATH = BASE_DIR / "data" / "grid" / "sensor_grid_map.csv"
OUTPUT_PATH = BASE_DIR / "data" / "output" / "gangnam_sewer_grid.csv"

# 현재 시각 기준으로 최근 1시간 조회 범위를 반환
def get_current_hour_range():
    now = datetime.now()
    start = (now - timedelta(hours=1)).strftime("%Y%m%d%H")
    end = now.strftime("%Y%m%d%H")
    return start, end

# 서울 Open API에서 강남구 하수관 수위 데이터를 조회
def get_gangnam_drainpipe_data(api_key):
    district_code = "23" #강남구 코드
    start_time, end_time = get_current_hour_range()

    url = (
        f"http://openAPI.seoul.go.kr:8088/"
        f"{api_key}/json/DrainpipeMonitoringInfo/1/1000/"
        f"{district_code}/{start_time}/{end_time}"
    )

    print("호출 URL:", url)

    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()

    print("API 응답 최상위 키:", data.keys())

    if "RESULT" in data:
        print("RESULT 내용:", data["RESULT"])

    rows = data.get("DrainpipeMonitoringInfo", {}).get("row", [])
    print("가져온 row 수:", len(rows))

    result = []
    for r in rows:
        result.append({
            "sensor_id": r.get("UNQ_NO"),
            "water_level": r.get("MSRMT_WATL"),
            "time": r.get("MSRMT_YMD"),
            "location": r.get("PSTN_INFO")
        })

    return result

# 수집한 센서 데이터에 grid_id를 매핑하고 CSV 파일로 저장
def attach_grid_and_save(rows):
    if not rows:
        print("저장할 새 row 없음")
        return pd.DataFrame(columns=["time", "grid_id", "water_level"])

    print("GRID_MAP_PATH:", GRID_MAP_PATH)
    print("OUTPUT_PATH:", OUTPUT_PATH)

    api_df = pd.DataFrame(rows)
    grid_map_df = pd.read_csv(GRID_MAP_PATH)

    api_df["sensor_id"] = api_df["sensor_id"].astype(str).str.strip()
    grid_map_df["sensor_id"] = grid_map_df["sensor_id"].astype(str).str.strip()

    merged = api_df.merge(
        grid_map_df[["sensor_id", "grid_id"]],
        on="sensor_id",
        how="left"
    )

    unmatched = merged[merged["grid_id"].isna()]
    if not unmatched.empty:
        print("매핑 안 된 센서:")
        print(unmatched[["sensor_id", "location"]].drop_duplicates())

    result = merged[["time", "grid_id", "water_level"]].copy()
    result["grid_id"] = result["grid_id"].astype("Int64")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    file_exists = OUTPUT_PATH.exists()
    result.to_csv(
        OUTPUT_PATH,
        mode="a",
        header=not file_exists,
        index=False,
        encoding="utf-8-sig"
    )

    print("파일 저장 완료:", OUTPUT_PATH)

    return result

# 일정 주기마다 API를 호출해 새 데이터만 저장
def run_polling(api_key, interval_seconds=10):
    while True:
        try:
            data = get_gangnam_drainpipe_data(api_key)
            new_rows = []

            for row in data:
                key = (row["sensor_id"], row["time"])
                if key not in seen:
                    seen.add(key)
                    new_rows.append(row)

            print("new_rows 개수:", len(new_rows))

            result_df = attach_grid_and_save(new_rows)

            print(f"[{datetime.now()}] 새 데이터: {len(new_rows)}건")
            if not result_df.empty:
                print(result_df.head())

        except Exception as e:
            print(f"[{datetime.now()}] 오류 발생: {e}")

        time.sleep(interval_seconds)


API_KEY = "api key"
run_polling(API_KEY, interval_seconds=300)


"""
센서 위치 뽑는 디버깅 코드
data = get_gangnam_drainpipe_data(API_KEY)

import pandas as pd
api_df = pd.DataFrame(data)

sensor_meta = api_df[["sensor_id", "location"]].drop_duplicates().sort_values("sensor_id")
print(sensor_meta)

sensor_meta.to_csv("gangnam_sensor_list.csv", index=False, encoding="utf-8-sig")
print("저장 완료: gangnam_sensor_list.csv")
"""