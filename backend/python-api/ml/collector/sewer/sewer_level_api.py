import time
import requests
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

seen = set()

BASE_DIR = Path(__file__).resolve().parents[3]
GRID_MAP_PATH = BASE_DIR / "data" / "grid" / "sensor_grid_map.csv"
OUTPUT_PATH = BASE_DIR / "data" / "output" / "seoul_sewer_api.parquet"

DISTRICT_CODES = [f"{i:02d}" for i in range(1, 26)]


# 현재 시각 기준으로 최근 1시간 조회 범위를 반환
def get_current_hour_range():
    now = datetime.now()
    start = (now - timedelta(hours=1)).strftime("%Y%m%d%H")
    end = now.strftime("%Y%m%d%H")
    return start, end


# 구별 하수관 수위 데이터 조회
def get_drainpipe_data_by_district(api_key, district_code):
    start_time, end_time = get_current_hour_range()

    url = (
        f"http://openAPI.seoul.go.kr:8088/"
        f"{api_key}/json/DrainpipeMonitoringInfo/1/1000/"
        f"{district_code}/{start_time}/{end_time}"
    )

    print("호출 URL:", url)

    response = requests.get(url, timeout=10)
    response.raise_for_status()

    try:
        data = response.json()
    except Exception:
        print("JSON 파싱 실패")
        print("status_code:", response.status_code)
        print("response text:", response.text[:500])
        raise

    print("API 응답 최상위 키:", data.keys())

    if "RESULT" in data:
        print("RESULT 내용:", data["RESULT"])

    rows = data.get("DrainpipeMonitoringInfo", {}).get("row", [])
    print(f"{district_code} 가져온 row 수:", len(rows))

    result = []
    for r in rows:
        result.append({
            "sensor_id": r.get("UNQ_NO"),
            "water_level": r.get("MSRMT_WATL"),
            "time": r.get("MSRMT_YMD"),
            "location": r.get("PSTN_INFO"),
        })

    return result


# 서울 전체(25개 구 순회) 조회
def get_seoul_drainpipe_data(api_key):
    all_rows = []

    for district_code in DISTRICT_CODES:
        try:
            rows = get_drainpipe_data_by_district(api_key, district_code)
            all_rows.extend(rows)
        except Exception as e:
            print(f"{district_code} 호출 실패: {e}")

    return all_rows


# 수집한 센서 데이터에 grid_id를 매핑하고 parquet 파일로 저장
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

    result = result.dropna(subset=["grid_id"])
    result["grid_id"] = result["grid_id"].astype("Int64")
    result["water_level"] = pd.to_numeric(
        result["water_level"], errors="coerce"
    ).astype("float32")
    result["time"] = pd.to_datetime(result["time"], errors="coerce").dt.floor("h")

    result = result.dropna(subset=["time", "water_level"])

    # 같은 시간 + 같은 grid면 최대 수위만 사용
    result = (
        result.groupby(["time", "grid_id"], as_index=False)["water_level"]
        .max()
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 기존 파일 있으면 concat
    if OUTPUT_PATH.exists():
        existing = pd.read_parquet(OUTPUT_PATH)
        result = pd.concat([existing, result], ignore_index=True)

        # 중복 제거
        result = result.sort_values(["time", "grid_id"])
        result = result.drop_duplicates(["time", "grid_id"], keep="last")

    # 저장
    result.to_parquet(OUTPUT_PATH, index=False)
    result.to_csv(OUTPUT_PATH.with_suffix(".csv"), index=False, encoding="utf-8-sig")

    print("Parquet 저장 완료:", OUTPUT_PATH)
    print("CSV 저장 완료:", OUTPUT_PATH.with_suffix(".csv"))

    return result


# 일정 주기마다 API를 호출해 새 데이터만 저장
def run_polling(api_key, interval_seconds=300):
    while True:
        try:
            data = get_seoul_drainpipe_data(api_key)
            new_rows = []

            for row in data:
                key = (str(row["sensor_id"]).strip(), row["time"])
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