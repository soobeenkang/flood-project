"""
공통 설정: 경로, 피처 정의, 타겟 등.
build_features.py와 train.py가 모두 import.
"""

# ── 경로 ─────────────────────────────────────────
RAW_DATA_DIR = "data/final_by_year_merged"   # 원본 final_YYYY.parquet
RAW_PATTERN = "final_*.parquet"

CACHE_DIR = "data/cache_features3"            # 피처 엔지니어링 결과 캐시
CACHE_PREFIX = "feat_"                        # feat_2010.parquet, feat_2011.parquet ...

MODEL_PATH = "flood_lgbm_model2.txt"           # 학습된 모델 저장 경로
BEST_PARAMS_PATH = "best_params2.json"       # Optuna 결과 저장 경로

# ── 캐시 추가 필터링 설정 ────────────────────────
FILTER_YEARS = [2023, 2024]  # 필터링을 적용할 연도
RAIN_FILTER_THRESHOLD = 5.0  # 침수 없을 때의 최소 강수량 조건 (5mm)

# ── 원본에서 로드할 컬럼 ─────────────────────────
RAW_COLS = [
    "grid_id", "time",
    "rain_1h", "rain_3h", "rain_6h", "rain_12h", "rain_24h",
    "rain_intensity", "rain_max_3h",
    "flood", "water_level", "mean_elevation", "is_river",
]


# ── 학습에 사용할 피처 ──────────────────────────
FEATURE_COLS = [
    # 원본 강수
    "rain_1h", "rain_3h", "rain_6h", "rain_12h", "rain_24h",
    "rain_intensity", "rain_max_3h",
    # 엔지니어링
    "rain_diff_1h", "rain_diff_3h", "rain_accel",
    "rain_ratio_1_24", "rain_ratio_3_12", "rain_1h_ma3",
    "rain_cat",
    # 지형
    "mean_elevation", "is_river",
    "topo_rain_risk", "elev_rain_12h",
    # 수위 (lag)
    # "water_level_lag1",
    # [추가] 과거 침수 기록 시계열 피처
    "is_flooded_lag1",          # 직전 시간 침수 여부
    "flood_sum_3h",        # 최근 3시간 누적 침수 시간
    "flood_duration_h",    # 연속 침수 지속 시간
    # 시간 (datetime은 정수 분해해서 사용)
    "hour_sin", "hour_cos",
]
TARGET_COL = "flood"

# train.py에서 캐시 로드할 때 필요한 컬럼만
LOAD_COLS = FEATURE_COLS + [TARGET_COL]


# ── 분할 / 학습 설정 ────────────────────────────
TEST_YEARS = 2          # 마지막 N개 연도를 test로
N_TRIALS = 5           # Optuna 시도 횟수
CV_FOLDS = 5            # TimeSeriesSplit fold 수
RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.5

# config.py 맨 아래 추가
RAIN_FILTER_THRESHOLD = 5.0  # 침수가 없는 시간대의 최소 1시간 강수량 기준 (mm)