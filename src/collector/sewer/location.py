import time
import requests
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import re


BASE_DIR = Path(__file__).resolve().parents[3]
OUTPUT_PATH = BASE_DIR / "data" / "sensor" / "sensor_locations.csv"

#서울 25개 구 코드
DISTRICT_CODES = [f"{i:02d}" for i in range(1, 26)]

#카카오 API 키 (REST API KEY)
KAKAO_API_KEY = "api key"

seen = set()

#최근 1시간 데이터 범위 생성
def get_current_hour_range():
    now = datetime.now()
    start = (now - timedelta(hours=1)).strftime("%Y%m%d%H")
    end = now.strftime("%Y%m%d%H")
    return start, end


#특정 구의 하수관 센서 데이터 가져오기
def get_drainpipe_data_by_district(api_key, district_code):
    start_time, end_time = get_current_hour_range()

    url = (
        f"http://openAPI.seoul.go.kr:8088/"
        f"{api_key}/json/DrainpipeMonitoringInfo/1/1000/"
        f"{district_code}/{start_time}/{end_time}"
    )

    print("호출 URL:", url)

    response = requests.get(url, timeout=10)
    print("status_code:", response.status_code)
    print("response text:", response.text[:300])

    response.raise_for_status()

    try:
        data = response.json()
    except Exception:
        print(f"{district_code} JSON 파싱 실패")
        raise

    rows = data.get("DrainpipeMonitoringInfo", {}).get("row", [])
    print(f"{district_code} 가져온 row 수:", len(rows))

    result = []
    for r in rows:
        result.append({
            "sensor_id": str(r.get("UNQ_NO")).strip(),
            "location": str(r.get("PSTN_INFO")).strip()
        })

    return result

#서울 전체 센서 데이터 수집
def get_seoul_sensor_data(api_key):
    all_rows = []

    #25개 구 반복
    for district_code in DISTRICT_CODES:
        try:
            rows = get_drainpipe_data_by_district(api_key, district_code)
            all_rows.extend(rows)
        except Exception as e:
            print(f"{district_code} 실패: {e}")

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["sensor_id"])
    return df

#주소 문자열 정제
def clean_address(addr: str) -> str:
    if not addr or str(addr).strip().lower() == "none":
        return ""

    addr = str(addr).strip()

    # HTML escape 제거
    addr = addr.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")

    # 꺾쇠/괄호 안 설명 제거
    addr = re.sub(r"<.*?>", "", addr)
    addr = re.sub(r"\(.*?\)", "", addr)

    # 대표적인 설명성 단어 뒤는 잘라냄
    addr = re.split(r"(앞|옆|뒤|사거리|교차로|맨홀|측구|지점에 위치|도로에 위치)", addr)[0]

    # 쉼표 뒤 설명 제거
    addr = addr.split(",")[0]

    # 다중 공백 정리
    addr = re.sub(r"\s+", " ", addr).strip()

    return addr

# 주소 -> 좌표 변환 (카카오 API)
def geocode_address(address):
    if not address:
        return None, None

    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    params = {"query": address}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        """
        디버깅 코드
        print("geocode query:", address)
        print("geocode status:", response.status_code)
        print("geocode text:", response.text[:300])

        """
        response.raise_for_status()
        result = response.json()

        documents = result.get("documents", [])
        if documents:
            x = documents[0]["x"]  # lon
            y = documents[0]["y"]  # lat
            return float(y), float(x)

    except Exception as e:
        print("geocode 실패:", address, e)

    return None, None

#센서 위치 데이터 업데이트 + 저장
def update_sensor_locations(api_key):
    new_df = get_seoul_sensor_data(api_key)

    if OUTPUT_PATH.exists():
        existing_df = pd.read_csv(OUTPUT_PATH)

        existing_df["sensor_id"] = existing_df["sensor_id"].astype(str).str.strip()

    else:
        existing_df = pd.DataFrame(columns=["sensor_id", "location", "lat", "lon"])

    # 기존 + 신규 합치기
    merged = pd.concat([existing_df, new_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=["sensor_id"], keep="first")

    # lat/lon 없는 애들만 geocoding
    for idx, row in merged.iterrows():
        if pd.isna(row.get("lat")) or pd.isna(row.get("lon")):
            cleaned = clean_address(row["location"])
            lat, lon = geocode_address(cleaned)
            merged.at[idx, "lat"] = lat
            merged.at[idx, "lon"] = lon

            #디버깅코드 : print(f"좌표 변환: {row['sensor_id']} → {lat}, {lon}")
            time.sleep(0.2)  # API rate 제한 방지

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\n=== sensor_locations.csv 저장 완료 ===")
    print("총 센서:", len(merged))
    print("좌표 없는 센서:", merged["lat"].isna().sum())

    return merged


if __name__ == "__main__":
    API_KEY = "api key"
    update_sensor_locations(API_KEY)