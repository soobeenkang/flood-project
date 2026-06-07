import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.preprocessing.data_xg import (
    GRID_COL,
    TIME_COL,
    RAIN_COL,
    WATER_COL,
    CUM_RAIN_COLS,
    EXTRA_RAIN_FEATURE_COLS,
    STATIC_FEATURE_COLS,
    RAIN_LAGS,
    CUM_RAIN_LAGS,
    EXTRA_RAIN_LAGS,
    WATER_LAGS,
    ROLL_WINDOWS,
    RAIN_THRESHOLD,
    preprocess_chunk,
    add_lag_delta_features,
    add_rolling_features,
)

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "model"
FEATURE_COLS_PATH = MODEL_DIR / "feature_cols.json"


def load_feature_cols():
    with open(FEATURE_COLS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


FEATURE_COLS = load_feature_cols()


def build_features_for_api(raw_data) -> pd.DataFrame:
    df = pd.DataFrame(raw_data).copy()

    required_base_cols = [
        GRID_COL,
        TIME_COL,
        RAIN_COL,
        WATER_COL,
        *CUM_RAIN_COLS,
        *EXTRA_RAIN_FEATURE_COLS,
        *STATIC_FEATURE_COLS,
    ]

    missing_base = [c for c in required_base_cols if c not in df.columns]
    if missing_base:
        raise ValueError(f"기본 입력 컬럼 누락: {missing_base}")

    df = preprocess_chunk(
        df,
        numeric_zero_fill_cols=[
            RAIN_COL,
            *CUM_RAIN_COLS,
            *EXTRA_RAIN_FEATURE_COLS,
        ],
    )

    if df.empty:
        raise ValueError("전처리 후 데이터가 비었습니다. time/month 범위를 확인하세요.")

    df = df.sort_values([GRID_COL, TIME_COL]).reset_index(drop=True)

    df["is_rainy_event"] = (df[RAIN_COL] >= RAIN_THRESHOLD).astype(np.int8)

    df = add_lag_delta_features(df, RAIN_COL, RAIN_LAGS)
    df = add_rolling_features(df, RAIN_COL, ROLL_WINDOWS)

    for col in CUM_RAIN_COLS:
        df = add_lag_delta_features(df, col, CUM_RAIN_LAGS)

    for col in EXTRA_RAIN_FEATURE_COLS:
        df = add_lag_delta_features(df, col, EXTRA_RAIN_LAGS)
        df = add_rolling_features(df, col, ROLL_WINDOWS)

    df = add_lag_delta_features(df, WATER_COL, WATER_LAGS)
    df = add_rolling_features(df, WATER_COL, ROLL_WINDOWS)

    df["hour"] = df[TIME_COL].dt.hour.astype(np.int8)
    df["dayofweek"] = df[TIME_COL].dt.dayofweek.astype(np.int8)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).astype(np.float32)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).astype(np.float32)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).astype(np.float32)

    # 없으면 임시 0 처리
    # 추후 학습용 grid_risk_score 매핑 파일 있으면 여기서 merge하는 게 정확함
    if "grid_risk_score" not in df.columns:
        df["grid_risk_score"] = 0.0

    # 실시간 예측은 보통 각 grid의 최신 시점만 예측
    df = (
        df.sort_values([GRID_COL, TIME_COL])
        .groupby(GRID_COL, as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"모델 입력 피처가 부족합니다. 누락 컬럼: {missing_cols}")

    feature_df = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)

    return feature_df