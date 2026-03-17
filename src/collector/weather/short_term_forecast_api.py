import requests
from datetime import datetime, timedelta
import math
import time

# 위경도 → 격자 변환

def latlon_to_grid(lat, lon):
    RE = 6371.00877
    GRID = 5.0
    SLAT1 = 30.0
    SLAT2 = 60.0
    OLON = 126.0
    OLAT = 38.0
    XO = 43
    YO = 136

    DEGRAD = math.pi / 180.0
    re = RE / GRID
    slat1 = SLAT1 * DEGRAD
    slat2 = SLAT2 * DEGRAD
    olon  = OLON  * DEGRAD
    olat  = OLAT  * DEGRAD

    sn = math.tan(math.pi * 0.25 + slat2 * 0.5) / math.tan(math.pi * 0.25 + slat1 * 0.5)
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(sn)
    sf = math.tan(math.pi * 0.25 + slat1 * 0.5)
    sf = (sf ** sn) * math.cos(slat1) / sn
    ro = math.tan(math.pi * 0.25 + olat * 0.5)
    ro = re * sf / (ro ** sn)
    ra = math.tan(math.pi * 0.25 + lat * DEGRAD * 0.5)
    ra = re * sf / (ra ** sn)
    theta = lon * DEGRAD - olon
    if theta > math.pi:  theta -= 2.0 * math.pi
    if theta < -math.pi: theta += 2.0 * math.pi
    theta *= sn

    x = int(ra * math.sin(theta) + XO + 0.5)
    y = int(ro - ra * math.cos(theta) + YO + 0.5)
    return x, y


# 격자 좌표 → 응답 배열 인덱스 변환
NX_TOTAL = 149
NY_TOTAL = 253

def grid_to_index(nx, ny):
    """1-based nx, ny → 0-based list index"""
    return (ny - 1) * NX_TOTAL + (nx - 1)


# 강남구 격자 좌표
GANGNAM_LAT = 37.5172
GANGNAM_LON = 127.0473
NX, NY = latlon_to_grid(GANGNAM_LAT, GANGNAM_LON)
TARGET_IDX = grid_to_index(NX, NY)
print(f"강남구 격자 좌표: NX={NX}, NY={NY}, 배열 인덱스={TARGET_IDX}")


# 발표시간 계산
# 초단기예보: 매 10분 발표, 발효시간은 +1h~+6h, 최소 20분 이전 발표시간 사용
def get_forecast_times(base_dt: datetime):
    adjusted = base_dt - timedelta(minutes=20)
    minute = (adjusted.minute // 10) * 10
    tmfc_dt = adjusted.replace(minute=minute, second=0, microsecond=0)
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
    for token in text.replace('\n', ',').split(','):
        token = token.strip()
        if token:
            try:
                values.append(float(token))
            except ValueError:
                pass
    return values


BASE_URL = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-dfs_vsrt_grd"
AUTH_KEY  = ""

SKY_CODE = {1: "맑음", 3: "구름많음", 4: "흐림"}
PTY_CODE = {0: "없음", 1: "비", 2: "비/눈", 3: "눈", 5: "빗방울", 6: "빗방울눈날림", 7: "눈날림"}


def fetch_value(tmfc: str, tmef: str, var: str) -> float | None:
    """단일 변수 API 호출 → 강남구 해당 인덱스 값 반환"""
    params = {
        "tmfc":    tmfc,
        "tmef":    tmef,
        "vars":    var,
        "authKey": AUTH_KEY,
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


def main():
    now = datetime.now()
    tmfc, tmfc_dt, tmef_list = get_forecast_times(now)

    print("=" * 62)
    print(f"  기상청 초단기예보 - 서울 강남구 날씨")
    print(f"  현재시각(KST): {now.strftime('%Y-%m-%d %H:%M')}")
    print(f"  발표시간(KST): {tmfc_dt.strftime('%Y-%m-%d %H:%M')}")
    print(f"  격자 좌표: NX={NX}, NY={NY}  (인덱스={TARGET_IDX})")
    print("=" * 62)
    print(f"{'시각(KST)':<14} {'기온(°C)':<10} {'하늘상태':<12} {'강수형태':<12} {'강수량(mm)'}")
    print("-" * 62)

    for tmef_dt, tmef in tmef_list:
        dt_str = tmef_dt.strftime("%m-%d %H시")

        t1h = fetch_value(tmfc, tmef, "T1H")
        sky = fetch_value(tmfc, tmef, "SKY")
        pty = fetch_value(tmfc, tmef, "PTY")
        rn1 = fetch_value(tmfc, tmef, "RN1")
        time.sleep(0.3)  # API 요청 간 간격

        t1h_str = f"{t1h:.1f}" if t1h is not None else "N/A"
        sky_str = SKY_CODE.get(int(sky), str(int(sky))) if sky is not None else "N/A"
        pty_str = PTY_CODE.get(int(pty), str(int(pty))) if pty is not None else "N/A"
        rn1_str = f"{rn1:.1f}" if rn1 is not None else "N/A"

        print(f"{dt_str:<14} {t1h_str:<10} {sky_str:<12} {pty_str:<12} {rn1_str}")

    print("=" * 62)
    print("* N/A: 비관측 영역이거나 아직 데이터 미제공")


if __name__ == "__main__":
    main()
