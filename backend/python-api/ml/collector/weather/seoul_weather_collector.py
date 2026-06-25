"""
서울시 기상청 날씨 수집기 (실황 + 초단기예보 통합)

!!!! 실행 전 사용자 설정 상수 입력하기 !!!!

■ 사용 API
  - 실황  (odam): nph-dfs_odam_grd
  - 초단기(vsrt): nph-dfs_vsrt_grd

■ 출력 컬럼
  grid_id    : int32          서울 100m 격자 ID
  tmfc       : datetime64[us] 발표시각

  [실황 — 현재]
  rn1_now    : float32   현재 강수량 (mm)
  t1h_now    : float32   현재 기온 (℃)
  vec_now    : float32   현재 풍향 (°)
  wsd_now    : float32   현재 풍속 (m/s)

  [초단기예보 — 강수량 +1h~+6h]
  rn1_1h ~ rn1_6h : float32  시간대별 예측 강수량 (mm)

  [초단기예보 — 하늘상태 +1h]
  sky_1h     : float32   1시간 후 하늘상태 (1=맑음, 2=구름조금, 3=구름많음, 4=흐림)

■ API 호출 횟수 / 사이클 : 실황 4회 + 초단기 7회 = 총 11회
"""

import json
import math
import time
import logging
import sys
from datetime import datetime, timedelta
from collections import deque
from sqlalchemy import dialects
from sqlalchemy.dialects.postgresql import insert

import os
from dotenv import load_dotenv

import requests
import pandas as pd
from pathlib import Path


# ════════════════════════════════════════════════
#  사용자 설정 상수
# ════════════════════════════════════════════════
PYTHON_API_DIR = Path("/app") 
PROJECT_ROOT = Path("/app") # 도커 컴포즈에서 env_file을 썼다면 필요 없을 수 있음

# .env 로드 (도커 환경변수가 우선이지만 하위 호환용)
load_dotenv(PROJECT_ROOT / ".env")
AUTH_KEY = os.getenv("WEATHER_AUTH_KEY")


if not AUTH_KEY:
    raise ValueError("WEATHER_AUTH_KEY가 .env에 설정되지 않았습니다.")
GEOJSON_PATH = PROJECT_ROOT / "data" / "seoul_grid.geojson"
DB_DIR_PATH = PYTHON_API_DIR / "seeds"
INTERVAL     = 3600 # 수집 주기(sec)


# ════════════════════════════════════════════════
#  db.py 동적 경로 추가 및 임포트
# ════════════════════════════════════════════════
if str(DB_DIR_PATH) not in sys.path:
    sys.path.append(str(DB_DIR_PATH))
print("DB_DIR_PATH:", DB_DIR_PATH)
print("sys.path added:", str(DB_DIR_PATH) in sys.path)
try:
    from db import get_engine

except ImportError as e:
    print(f"[오류] db.py를 찾을 수 없습니다. 에러: {e}")
    sys.exit(1)

# ════════════════════════════════════════════════
# 로깅 설정
# ════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DB_TABLE = "seoul_weather"

# ════════════════════════════════════════════════
# 1. Lambert Conformal Conic 투영 변환
# ════════════════════════════════════════════════

class LCCConverter:
    """
    위경도(WGS84) → 기상청 LCC 격자 좌표 변환기

    기상청 격자 파라미터 (동네예보 격자영역 정보 PDF 기준)
    - 지구반경   : 6371.00877 km
    - 표준위도   : 30.0N, 60.0N
    - 기준경도   : 126.0E  /  기준위도 : 38.0N
    - 기준격자   : X0=43(동서), Y0=136(남북)
    - 격자간격   : 5 km  /  격자수 : 149 × 253
    """
    RE = 6371.00877; GRID = 5.0
    SLAT1 = 30.0;    SLAT2 = 60.0
    OLON  = 126.0;   OLAT  = 38.0
    XO    = 43.0;    YO    = 136.0
    DEGRAD = math.pi / 180.0
    NX = 149;        NY = 253

    def __init__(self):
        re    = self.RE / self.GRID
        slat1 = self.SLAT1 * self.DEGRAD
        slat2 = self.SLAT2 * self.DEGRAD
        olon  = self.OLON  * self.DEGRAD
        olat  = self.OLAT  * self.DEGRAD
        sn = math.log(math.cos(slat1) / math.cos(slat2)) / \
             math.log(math.tan(math.pi * 0.25 + slat2 * 0.5) /
                      math.tan(math.pi * 0.25 + slat1 * 0.5))
        sf = (math.tan(math.pi * 0.25 + slat1 * 0.5) ** sn) * math.cos(slat1) / sn
        ro = re * sf / (math.tan(math.pi * 0.25 + olat * 0.5) ** sn)
        self.re = re; self.sn = sn; self.sf = sf; self.ro = ro; self.olon = olon

    def latlon_to_grid(self, lat: float, lon: float) -> tuple[int, int]:
        ra    = self.re * self.sf / (math.tan(math.pi * 0.25 + lat * self.DEGRAD * 0.5) ** self.sn)
        theta = (lon * self.DEGRAD - self.olon) * self.sn
        nx = int(ra * math.sin(theta) + self.XO + 0.5)
        ny = int(self.ro - ra * math.cos(theta) + self.YO + 0.5)
        return nx, ny

    @staticmethod
    def grid_to_index(nx: int, ny: int, grid_nx: int = 149) -> int:
        """(nx, ny) → 응답 배열 0-based 인덱스. 좌하단(1,1) 기준."""
        return (ny - 1) * grid_nx + (nx - 1)


# ════════════════════════════════════════════════
# 2. GeoJSON 로드 → 격자 매핑 테이블
# ════════════════════════════════════════════════

def build_grid_mapping(geojson_path: str, converter: LCCConverter) -> pd.DataFrame:
    """
    서울 100m 격자 GeoJSON → (grid_id, arr_idx) DataFrame.
    arr_idx : API 응답 배열에서 해당 격자의 0-based 위치
    """
    logger.info("GeoJSON 로딩: %s", geojson_path)
    with open(geojson_path, encoding="utf-8") as f:
        gj = json.load(f)

    rows = []
    for feat in gj["features"]:
        props  = feat["properties"]
        nx, ny = converter.latlon_to_grid(props["lat"], props["lon"])
        rows.append({
            "grid_id" : props["grid_id"],
            "arr_idx" : converter.grid_to_index(nx, ny),
        })

    df = pd.DataFrame(rows)
    logger.info("총 격자 수: %d", len(df))
    return df


# ════════════════════════════════════════════════
# 3. API 공통 설정 및 유틸리티
# ════════════════════════════════════════════════

URL_ODAM   = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-dfs_odam_grd"
URL_VSRT   = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-dfs_vsrt_grd"
KMA_NX     = 149
KMA_NY     = 253
FORECAST_HOURS = list(range(1, 7))   # +1h ~ +6h

LAG_MIN    = 20    # 발표 처리 지연 여유 (분)
TIMEOUT    = 120   # HTTP 응답 대기 (초)
MAX_RETRY  = 3     # 최대 재시도 횟수
RETRY_WAIT = 10    # 재시도 간격 (초)


def get_latest_tmfc() -> str:
    """
    현재 시각 기준 안전한 발표시각 반환 — yyyymmddhhmm.
    LAG_MIN(20분)을 뺀 뒤 10분 단위로 내림.
    """
    now = datetime.now()
    
    # 현재 분이 45분 미만이면 아직 이번 시간대 데이터가 생성되지 않았으므로 '1시간 전 40분' 데이터 타겟팅
    if now.minute < 45:
        target_time = (now - timedelta(hours=1)).replace(minute=40, second=0, microsecond=0)
    else:
        target_time = now.replace(minute=40, second=0, microsecond=0)

    return target_time.strftime("%Y%m%d%H%M")


def tmfc_to_tmef(tmfc: str, hours: int) -> str:
    """
    발표시각(tmfc, yyyymmddhhmm) + hours 시간 → 발효시각(tmef, yyyymmddhh).
    예) tmfc="202604041030", hours=2 → tmef="2026040412"
    """
    base = datetime.strptime(tmfc, "%Y%m%d%H%M")
    return (base + timedelta(hours=hours)).strftime("%Y%m%d%H")


def _parse_response(text: str) -> list:
    """
    API 응답 CSV 텍스트 → list[float|None] (길이 NX×NY).
    비관측(-99.0) → None으로 치환.
    """
    tokens = [t.strip() for t in text.replace("\n", ",").split(",") if t.strip()]
    values = []
    for tok in tokens:
        try:
            v = float(tok)
            values.append(None if v == -99.0 else v)
        except ValueError:
            continue

    expected = KMA_NX * KMA_NY
    if len(values) < expected:
        logger.warning("응답 값 부족: %d / %d → None 패딩", len(values), expected)
        values += [None] * (expected - len(values))
    elif len(values) > expected:
        values = values[:expected]
    return values


def _fetch(url: str, params: dict, label: str) -> list | None:
    """
    공통 HTTP 요청 함수. MAX_RETRY회 재시도.
    성공 시 list[float|None], 실패 시 None 반환.
    """
    for attempt in range(1, MAX_RETRY + 1):
        try:
            logger.info("요청 %d/%d — %s", attempt, MAX_RETRY, label)
            resp = requests.get(url, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            values      = _parse_response(resp.text)
            valid_count = sum(1 for v in values if v is not None)
            logger.info("수신 완료 — %s, 유효값: %d / %d",
                        label, valid_count, KMA_NX * KMA_NY)
            return values
        except requests.exceptions.Timeout:
            logger.warning("타임아웃 (시도 %d/%d) — %s", attempt, MAX_RETRY, label)
        except requests.exceptions.HTTPError as exc:
            logger.warning("HTTP 오류 (시도 %d/%d) — %s: %s", attempt, MAX_RETRY, label, exc)
        except Exception as exc:
            logger.warning("기타 오류 (시도 %d/%d) — %s: %s", attempt, MAX_RETRY, label, exc)

        if attempt < MAX_RETRY:
            logger.info("%.0f초 후 재시도...", RETRY_WAIT)
            time.sleep(RETRY_WAIT)

    logger.error("모든 재시도 실패 — %s", label)
    return None


# ════════════════════════════════════════════════
# 4. 실황 API 호출 (nph-dfs_odam_grd)
#    제공 변수: T1H, UUU, VVV, VEC, WSD, PTY, RN1, REH
# ════════════════════════════════════════════════

def fetch_odam(auth_key: str, tmfc: str, var: str) -> list | None:
    """
    실황(odam) 전체 격자 단일 변수 조회.

    Parameters
    ----------
    tmfc : 발표시각 yyyymmddhhmm (10분 단위)
    var  : 변수명 (T1H / VEC / WSD / RN1 등)
    """
    params = {"tmfc": tmfc, "vars": var, "authKey": auth_key}
    return _fetch(URL_ODAM, params, label=f"odam/{var} tmfc={tmfc}")


# ════════════════════════════════════════════════
# 5. 초단기예보 API 호출 (nph-dfs_vsrt_grd)
#    제공 변수: T1H, UUU, VVV, VEC, WSD, SKY, LGT, PTY, RN1, REH
# ════════════════════════════════════════════════

def fetch_vsrt(auth_key: str, tmfc: str, tmef: str, var: str) -> list | None:
    """
    초단기예보(vsrt) 전체 격자 단일 변수 조회.

    Parameters
    ----------
    tmfc : 발표시각 yyyymmddhhmm (10분 단위)
    tmef : 발효시각 yyyymmddhh   (tmfc 기준 +1h~+6h)
    var  : 변수명 (RN1 / SKY / PTY 등)
    """
    params = {"tmfc": tmfc, "tmef": tmef, "vars": var, "authKey": auth_key}
    return _fetch(URL_VSRT, params, label=f"vsrt/{var} tmfc={tmfc} tmef={tmef}")


# ════════════════════════════════════════════════
# 6. 슬라이딩 윈도우 — 실황 RN1 최근 10개 유지
# ════════════════════════════════════════════════

class GridWindow:
    """
    실황 RN1 전체 격자 배열을 최근 10개 시간대 보관.
    deque(maxlen=10): 11번째 push 시 가장 오래된 스냅샷 자동 제거.
    """
    MAXLEN  = 10
    GRIDLEN = KMA_NX * KMA_NY

    def __init__(self):
        self._window: deque = deque(maxlen=self.MAXLEN)

    def push(self, grid_values: list | None):
        """새 스냅샷 추가. None이면 전부 None 배열로 대체."""
        self._window.append(
            grid_values if grid_values is not None
            else [None] * self.GRIDLEN
        )

    def _val_at(self, snapshot: list, idx: int) -> float | None:
        if idx < 0 or idx >= len(snapshot):
            return None
        return snapshot[idx]

    def current(self, idx: int) -> float | None:
        """현재(최신 스냅샷) 값."""
        if not self._window:
            return None
        return self._val_at(self._window[-1], idx)


# ════════════════════════════════════════════════
# 7. 단일 격자 인덱스에서 값 추출 헬퍼
# ════════════════════════════════════════════════

def extract(grid: list | None, idx: int) -> float | None:
    """
    전체 격자 배열(grid)에서 idx 위치 값 반환.
    grid가 None이거나 idx 범위 초과이면 None.
    """
    if grid is None:
        return None
    if idx < 0 or idx >= len(grid):
        return None
    return grid[idx]

def insert_on_conflict(table, conn, keys, data_iter):
    data = [dict(zip(keys, row)) for row in data_iter]
    stmt = insert(table.table).values(data)
    
    # 충돌 발생 시 업데이트할 컬럼들 목록 세팅 (grid_id, tmfc 제외한 나머지 변수들)
    update_dict = {c.name: c for c in stmt.excluded if c.name not in ['grid_id', 'tmfc']}
    
    on_conflict_stmt = stmt.on_conflict_do_update(
        index_elements=['grid_id', 'tmfc'],
        set_=update_dict
    )
    conn.execute(on_conflict_stmt)

# ════════════════════════════════════════════════
# 8. 한 사이클: API 호출 → db 저장
# ════════════════════════════════════════════════

def run_cycle(
    grid_mapping : pd.DataFrame,
    window       : GridWindow,
    auth_key     : str,
    tmfc         : str | None = None,
):
    """
    1회 수집 사이클 (총 API 호출 11회).

    [실황 odam — 4회]
      RN1  → rn1_now  (현재 강수량, GridWindow에도 push)
      T1H  → t1h_now  (현재 기온)
      VEC  → vec_now  (현재 풍향)
      WSD  → wsd_now  (현재 풍속)

    [초단기예보 vsrt — 7회]
      RN1 × +1h~+6h → rn1_1h ~ rn1_6h  (강수량 예측)
      SKY × +1h     → sky_1h            (하늘상태 예측)
    """
    if tmfc is None:
        tmfc = get_latest_tmfc()

    logger.info("═" * 50)
    logger.info("수집 사이클 시작 — tmfc=%s", tmfc)

    # ── 8-1. 실황 조회 (4변수) ───────────────────
    odam_rn1 = fetch_odam(auth_key, tmfc, var="RN1")
    odam_t1h = fetch_odam(auth_key, tmfc, var="T1H")
    odam_vec = fetch_odam(auth_key, tmfc, var="VEC")
    odam_wsd = fetch_odam(auth_key, tmfc, var="WSD")
    if odam_rn1 is None:
        logger.error("[날씨 api] 기상청 실황 rn1 데이터 가져오지 못함. 사이클 스킵")
        return None
    window.push(odam_rn1)   # RN1만 슬라이딩 윈도우에 유지

    # ── 8-2. 초단기예보 조회 ─────────────────────
    # 강수량: +1h ~ +6h
    vsrt_rn1: dict[int, list | None] = {}
    for h in FORECAST_HOURS:
        tmef = tmfc_to_tmef(tmfc, h)
        vsrt_rn1[h] = fetch_vsrt(auth_key, tmfc, tmef, var="RN1")

    # 하늘상태: +1h 만
    tmef_1h  = tmfc_to_tmef(tmfc, 1)
    vsrt_sky = fetch_vsrt(auth_key, tmfc, tmef_1h, var="SKY")

    # ── 8-3. 서울 격자별 값 매핑 ─────────────────
    arr_idx  = grid_mapping["arr_idx"].to_numpy()
    grid_ids = grid_mapping["grid_id"].to_numpy()
    tmfc_dt  = pd.to_datetime(tmfc, format="%Y%m%d%H%M")

    data = {
        # 식별자
        "grid_id"  : pd.array(grid_ids,  dtype="int32"),
        "tmfc"     : pd.array([tmfc_dt] * len(grid_ids), dtype="datetime64[us]"),
        # 현재 실황
        "rn1_now"  : pd.array([window.current(int(i))    for i in arr_idx], dtype="float32"),
        "t1h_now"  : pd.array([extract(odam_t1h, int(i)) for i in arr_idx], dtype="float32"),
        "vec_now"  : pd.array([extract(odam_vec, int(i)) for i in arr_idx], dtype="float32"),
        "wsd_now"  : pd.array([extract(odam_wsd, int(i)) for i in arr_idx], dtype="float32"),
        # 1시간 후 하늘상태
        "sky_1h"   : pd.array([extract(vsrt_sky, int(i)) for i in arr_idx], dtype="float32"),
    }

    # 강수량 예측 +1h ~ +6h
    for h in FORECAST_HOURS:
        data[f"rn1_{h}h"] = pd.array(
            [extract(vsrt_rn1[h], int(i)) for i in arr_idx], dtype="float32"
        )

    df_out = pd.DataFrame(data)

    # ── 8-4. db 저장 ────────────────────────
    engine = get_engine()
    with engine.begin() as conn:
        df_out.to_sql(
            name      = DB_TABLE,
            con       = conn,
            if_exists = "append",
            index     = False,
            method    = insert_on_conflict,
        )
    logger.info("DB 저장 완료 → 테이블: %s  (행: %d)", DB_TABLE, len(df_out))

    return df_out

# ════════════════════════════════════════════════
# 9. 메인 루프 — 1시간 간격 반복 수집
# ════════════════════════════════════════════════

# 외부에서 스케줄러가 매번 GeoJSON을 다시 읽지 않도록 미리 초기화해둡니다.
converter = LCCConverter()
grid_mapping = build_grid_mapping(GEOJSON_PATH, converter)
window = GridWindow()


def collect_weather_data():
    """
    APScheduler가 매 정각마다 호출할 단일 실행 함수입니다.
    기존에 1시간마다 무한 루프 돌던 부분을 1회성 사이클 함수로 연결합니다.
    """
    logger.info("⏰ [스케줄러 펑션] 날씨 API 정기 수집 가동")
    try:
        run_cycle(
            grid_mapping=grid_mapping,
            window=window,
            auth_key=AUTH_KEY,
        )
    except Exception as exc:
        logger.error("정기 수집 사이클 오류: %s", exc, exc_info=True)

'''
def main():
    converter    = LCCConverter()
    grid_mapping = build_grid_mapping(GEOJSON_PATH, converter)
    window       = GridWindow()

    logger.info("수집 시작 (주기: %d초)", INTERVAL)

    while True:
        try:
            run_cycle(
                grid_mapping = grid_mapping,
                window       = window,
                auth_key     = AUTH_KEY,
            )
        except Exception as exc:
            logger.error("사이클 오류: %s", exc, exc_info=True)

        logger.info("다음 수집까지 %d초 대기...", INTERVAL)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
'''