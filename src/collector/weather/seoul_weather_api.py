import requests
import json
import logging
import argparse

from collections import deque
from datetime import datetime, timedelta

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import math
import time

# 로그 출력 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 위경도 -> 기상청 격자 변환(Lambert conformal conic 투영 계산 공식) 
class LCCConverter:
    RE     = 6371.00877
    GRID   = 5.0
    SLAT1  = 30.0
    SLAT2  = 60.0
    OLON   = 126.0
    OLAT   = 38.0
    XO     = 43.0
    YO     = 136.0
    
    DEGRAD = math.pi / 180.0
    NX = 149   # 가로 격자 수
    NY = 253   # 세로 격자 수

    def __init__(self):
        re    = self.RE / self.GRID
        slat1 = self.SLAT1 * self.DEGRAD
        slat2 = self.SLAT2 * self.DEGRAD
        olon  = self.OLON  * self.DEGRAD
        olat  = self.OLAT  * self.DEGRAD

        sn = math.log(math.cos(slat1) / math.cos(slat2)) / \
            math.log(math.tan(math.pi * 0.25 + slat2 * 0.5) /
                math.tan(math.pi * 0.25 + slat1 * 0.5))
        sf = math.tan(math.pi * 0.25 + slat1 * 0.5)
        sf = (sf ** sn) * math.cos(slat1) / sn
        ro = math.tan(math.pi * 0.25 + olat * 0.5)
        ro = re * sf / (ro ** sn)

        self.re   = re
        self.sn   = sn
        self.sf   = sf
        self.ro   = ro
        self.olon = olon

    # 위경도 -> 기상청 격자
    def latlon_to_grid(self, lat: float, lon: float) -> tuple[int, int]:
        ra = math.tan(math.pi * 0.25 + lat * self.DEGRAD * 0.5)
        ra = self.re * self.sf / (ra ** self.sn)
        
        theta = lon * self.DEGRAD - self.olon
        if theta > math.pi: theta -= 2.0 * math.pi
        if theta < -math.pi: theta += 2.0 * math.pi
        theta *= self.sn

        nx = int(ra * math.sin(theta) + self.XO + 0.5)
        ny = int(self.ro - ra * math.cos(theta) + self.YO + 0.5)
        return nx, ny

    @staticmethod # 그냥 계산만 하니까 static으로..
    # 격자 -> 배열 인덱스
    def grid_to_index(nx: int, ny: int, grid_nx: int = 149) -> int:
        # 좌측하단(nx=1, ny=1)에서 시작, 동쪽(nx 증가) → 북쪽(ny 증가) 순으로 저장 
        return (ny - 1) * grid_nx + (nx - 1)


# geojson 읽고 데이터 구조에 격자 매핑 저장
def build_grid_mapping(geojson_path: str, converter: LCCConverter) -> pd.DataFrame:
    logger.info("GeoJSON 로딩: %s", geojson_path)
    with open(geojson_path, encoding="utf-8") as f:
        gj = json.load(f)

    rows = []
    for feat in gj["features"]:
        props  = feat["properties"]
        lat    = props["lat"]
        lon    = props["lon"]
        gid    = props["grid_id"]
        nx, ny = converter.latlon_to_grid(lat, lon) # 위경도 -> 기상청 격자
        idx    = converter.grid_to_index(nx, ny) # 배열 인덱스 계산
        rows.append({
            "grid_id" : gid,
            "lat"     : lat,
            "lon"     : lon,
            "kma_nx"  : nx,
            "kma_ny"  : ny,
            "arr_idx" : idx,
        })

    df = pd.DataFrame(rows)
    logger.info("총 격자 수: %d  /  고유 기상청 격자 수: %d",
                len(df), df[["kma_nx", "kma_ny"]].drop_duplicates().shape[0])
    return df


# 기상청 API 호출 및 파싱
KMA_URL    = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-dfs_odam_grd"
KMA_NX     = 149
KMA_NY     = 253  # 격자 크기
LAG_MIN    = 20   # 현재 시각보다 20분 이전 사용(데이터 지연)
TIMEOUT    = 120  # 응답 대기 시간(sec)
MAX_RETRY  = 3    # 최대 재시도 횟수
RETRY_WAIT = 10   # 재시도 대기(sec)


# 사용할 시간 계산(현재 시각에서 20분(LAG_MIN) 빼고 10분단위로 내림)
def get_latest_tmfc() -> str:
    now    = datetime.now()
    lagged = now - timedelta(minutes=LAG_MIN)
    minute = (lagged.minute // 10) * 10
    t      = lagged.replace(minute=minute, second=0, microsecond=0)
    return t.strftime("%Y%m%d%H%M")


# 응답 파싱
def _parse_response(text: str) -> list:
    tokens = [t.strip() for t in text.replace("\n", ",").split(",") if t.strip()]
    values = []
    for tok in tokens:
        try:
            v = float(tok)
            values.append(None if v == -99.0 else v)
        except ValueError:
            continue  # 숫자 아닌거 무시

    total = KMA_NX * KMA_NY #전체 기상청 격자 개수
    if len(values) < total:
        logger.warning("응답 값 개수 부족: %d (기대 %d), None으로 패딩",
                       len(values), total)
        values += [None] * (total - len(values))
    elif len(values) > total:
        logger.warning("응답 값 개수 초과: %d (기대 %d), 앞 %d개만 사용",
                       len(values), total, total)
        values = values[:total]
    return values


# API 호출, 전체 격자 데이터 한번에 파싱
def fetch_full_grid(auth_key: str, tmfc: str | None = None,
                    var: str = "RN1") -> list | None:
    if tmfc is None:
        tmfc = get_latest_tmfc()

    params = {
        "tmfc"   : tmfc,
        "vars"   : var,
        "authKey": auth_key,
    }

    for attempt in range(1, MAX_RETRY + 1):
        try:
            logger.info("API 요청 시도 %d/%d — tmfc=%s", attempt, MAX_RETRY, tmfc)
            resp = requests.get(KMA_URL, params=params, timeout=TIMEOUT)
            resp.raise_for_status()

            values      = _parse_response(resp.text)
            valid_count = sum(1 for v in values if v is not None)
            logger.info("전체 격자 수신 완료 — tmfc=%s, 유효값: %d / %d",
                        tmfc, valid_count, KMA_NX * KMA_NY)
            return values

        # 오류 발생의 경우
        # 10초 대기 후 3번까지 재시도(RETRY_WAIT = 10, MAX_RETRY = 3)
        except requests.exceptions.Timeout:
            logger.warning("타임아웃 (시도 %d/%d, timeout=%ds)",
                           attempt, MAX_RETRY, TIMEOUT)
        except requests.exceptions.HTTPError as exc:
            logger.warning("HTTP 오류 (시도 %d/%d): %s", attempt, MAX_RETRY, exc)
        except Exception as exc:
            logger.warning("기타 오류 (시도 %d/%d): %s", attempt, MAX_RETRY, exc)

        if attempt < MAX_RETRY:
            logger.info("%.0f초 후 재시도...", RETRY_WAIT)
            time.sleep(RETRY_WAIT)

    logger.error("모든 재시도 실패 — tmfc=%s, 해당 시간대는 None 처리", tmfc)
    return None


# 전체 격자 배열을 최근 10개 시간대 보관
# deque 구조 [t-9, t-8, ..., t0] (오래된게 앞 최신이 뒤)
# 10개 초과 시 가장 오래된 배열 자동 제거(maxlen = 10)
class GridWindow:
    MAXLEN  = 10
    GRIDLEN = KMA_NX * KMA_NY  # 37,697

    def __init__(self):
        self._window: deque = deque(maxlen=self.MAXLEN)
    
    # 새 시간대 배열 추가, 못 받아왔으면 None
    def push(self, grid_values: list | None):
        if grid_values is None:
            grid_values = [None] * self.GRIDLEN
        self._window.append(grid_values)

    # 특정 격자의 강수량 가져오기
    def _val_at(self, snapshot: list, idx: int) -> float | None:
        if idx < 0 or idx >= len(snapshot):
            return None
        return snapshot[idx]  # None 또는 float

    # n시간 누적 강수량값
    # 쌓인 값이 n개 미만            -> None,
    # n개 이상인데 값이 전부 None   -> None,
    # 일부만 None이면               -> None 아닌 것만 합산
    def acc_nh(self, idx: int, n: int) -> float | None:
        if len(self._window) < n:
            return None  # 쌓인 값이 n개 미만

        snaps  = list(self._window)[-n:]
        vals   = [self._val_at(s, idx) for s in snaps]
        valids = [v for v in vals if v is not None]

        if not valids:
            return None  # 전부 비관측

        return sum(valids)


# 한 번의 수집 과정(cycle)
# 1. 전체 격자 1회 호출
# 2. 저장소에 전체 격자 강수량 추가(최근 10개)
# 3. 그리드별 강수량 조회
# 4. 누적 강수량 계산
# 5. 파켓 파일로 저장
def run_cycle(
    grid_mapping : pd.DataFrame,
    window       : GridWindow,
    auth_key     : str,
    output_path  : str,
    tmfc         : str | None = None,
):

    if tmfc is None:
        tmfc = get_latest_tmfc()

    logger.info("수집 시작 — tmfc=%s", tmfc)

    # 전체 격자 강수량 가져오기
    grid_values = fetch_full_grid(auth_key, tmfc=tmfc, var="RN1") 
    window.push(grid_values) 

    # 그리드별 강수량 
    arr_idx  = grid_mapping["arr_idx"].to_numpy() # 서울 그리드 -> 기상청 격자 인덱스
    grid_ids = grid_mapping["grid_id"].to_numpy()

    # 1시간/3시간/6시간 누적 강수량
    acc_1h_vals  = [window.acc_nh(int(i), 1)  for i in arr_idx]
    acc_3h_vals  = [window.acc_nh(int(i), 3)  for i in arr_idx]
    acc_6h_vals  = [window.acc_nh(int(i), 6)  for i in arr_idx]

    # tmfc 문자열 -> datetime64[us] 변환
    tmfc_dt = pd.to_datetime(tmfc, format="%Y%m%d%H%M")

    # 최종 결과 테이블 생성
    df_out = pd.DataFrame({
        "grid_id" : pd.array(grid_ids,      dtype="int32"),
        "tmfc"    : pd.array([tmfc_dt] * len(grid_ids), dtype="datetime64[us]"),
        "acc_1h"  : pd.array(acc_1h_vals,   dtype="float32"),
        "acc_3h"  : pd.array(acc_3h_vals,   dtype="float32"),
        "acc_6h"  : pd.array(acc_6h_vals,   dtype="float32"),
    })

    # 파켓 형식으로 저장
    table = pa.Table.from_pandas(df_out, preserve_index=False)
    pq.write_table(table, output_path, compression="snappy")
    logger.info("Parquet 저장 완료 → %s  (행 수: %d, 컬럼: %s)",
                output_path, len(df_out), list(df_out.columns))

    return df_out



# 1시간 간격 반복 수집
def main():
    parser = argparse.ArgumentParser(description="서울시 기상청 실황 강수량 수집기")
    # 명령어 옵션으로 auth key 입력받기
    parser.add_argument("--auth-key", required=True,  help="기상청 API 인증키")
    # geojson 파일 경로 입력, 입력x시 디폴트로 seoul_grid.geojson
    parser.add_argument("--geojson",  default="seoul_grid.geojson",
                        help="서울 격자 GeoJSON 경로")
    # 출력 파켓 파일 관련 설정
    parser.add_argument("--output",   default="seoul_rain.parquet",
                        help="출력 Parquet 파일 경로")
    # 데이터 수집 주기, 기본 1시간
    parser.add_argument("--interval", type=int, default=3600,
                        help="수집 주기(초), 기본 3600 = 1시간")
    # 입력한 옵션 읽어서 args 변수에 저장
    args = parser.parse_args()

    converter    = LCCConverter()
    grid_mapping = build_grid_mapping(args.geojson, converter)
    window       = GridWindow()

    logger.info("수집 시작 (주기: %d초)", args.interval)

    while True:
        try:
            run_cycle(
                grid_mapping = grid_mapping,
                window       = window,
                auth_key     = args.auth_key,
                output_path  = args.output,
            )
        except Exception as exc:
            logger.error("사이클 오류: %s", exc, exc_info=True)

        logger.info("다음 수집까지 %d초 대기...", args.interval)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
