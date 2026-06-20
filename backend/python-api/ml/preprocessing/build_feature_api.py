import numpy as np
import pandas as pd

from ml.config import FEATURE_COLS


def build_features_for_api(raw_data) -> pd.DataFrame:
    df = pd.DataFrame(raw_data).copy()

    required_cols = [
        "grid_id", "time",
        "rain_1h", "rain_3h", "rain_6h", "rain_12h", "rain_24h",
        "rain_intensity", "rain_max_3h",
        "mean_elevation", "is_river",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"기본 입력 컬럼 누락: {missing}")

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(["grid_id", "time"]).reset_index(drop=True)

    numeric_cols = [
        "rain_1h", "rain_3h", "rain_6h", "rain_12h", "rain_24h",
        "rain_intensity", "rain_max_3h",
        "mean_elevation", "is_river",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    g = df.groupby("grid_id", observed=True, sort=False)

    df["rain_diff_1h"] = g["rain_1h"].diff().fillna(0)
    df["rain_diff_3h"] = g["rain_3h"].diff().fillna(0)
    df["rain_accel"] = g["rain_diff_1h"].diff().fillna(0)

    df["rain_ratio_1_24"] = df["rain_1h"] / (df["rain_24h"] + 1e-6)
    df["rain_ratio_3_12"] = df["rain_3h"] / (df["rain_12h"] + 1e-6)

    df["rain_1h_ma3"] = g["rain_1h"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

    bins = [-np.inf, 0, 2, 10, 30, np.inf]
    df["rain_cat"] = pd.cut(
        df["rain_intensity"],
        bins=bins,
        labels=[0, 1, 2, 3, 4],
    ).astype("float32")

    df["topo_rain_risk"] = (
        (1 / (df["mean_elevation"] + 1)) * df["rain_24h"] * df["is_river"]
    )
    df["elev_rain_12h"] = df["rain_12h"] / (df["mean_elevation"] + 1)

    # API 입력에 과거 flood가 있으면 사용, 없으면 0으로 처리
    if "flood" in df.columns:
        df["flood"] = pd.to_numeric(df["flood"], errors="coerce").fillna(0)
        df["is_flooded_lag1"] = g["flood"].shift(1).fillna(0).astype("int8")
    else:
        df["is_flooded_lag1"] = 0

    g_lag = df.groupby("grid_id", observed=True, sort=False)

    df["flood_sum_3h"] = (
        g_lag["is_flooded_lag1"]
        .transform(lambda x: x.rolling(3, min_periods=1).sum())
        .fillna(0)
        .astype("int8")
    )

    def calc_duration(series):
        group_ids = (series == 0).cumsum()
        return series.groupby(group_ids).cumsum()

    df["flood_duration_h"] = (
        g_lag["is_flooded_lag1"]
        .transform(calc_duration)
        .fillna(0)
        .astype("int16")
    )

    df["hour"] = df["time"].dt.hour.astype("int8")
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).astype("float32")

    # 실시간 예측은 grid별 최신 시점만 사용
    df = (
        df.sort_values(["grid_id", "time"])
        .groupby("grid_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    missing_features = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_features:
        raise ValueError(f"모델 입력 피처 누락: {missing_features}")

    feature_df = (
        df[FEATURE_COLS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype("float32")
    )

    return feature_df