import os
import glob
import gc
from collections import defaultdict

import numpy as np
import pandas as pd


# =========================================================
# 0. 설정
# =========================================================
RAIN_DIR = r"data/tmp_final"
OUTPUT_PATH = r"data/final/event_sampled_dataset_2pass.parquet"

GRID_COL = "grid_id"
TIME_COL = "time"
YEAR_COL = "year"
MONTH_COL = "month"

TARGET_COL = "flood"

RAIN_COL = "rain_1h"
CUM_RAIN_COLS = ["rain_3h", "rain_6h", "rain_12h", "rain_24h"]
EXTRA_RAIN_FEATURE_COLS = ["rain_intensity", "rain_max_3h"]
WATER_COL = "water_level"
STATIC_FEATURE_COLS = ["mean_elevation", "is_river"]

RAIN_THRESHOLD = 0.1
NEG_POS_RATIO = 5

RAIN_LAGS = [1, 2, 3, 6, 12, 24]
CUM_RAIN_LAGS = [1, 2, 3, 6, 12]
EXTRA_RAIN_LAGS = [1, 2, 3, 6, 12]
WATER_LAGS = [1, 2, 3, 6, 12, 24]
ROLL_WINDOWS = [3, 6, 12]

# 파일 경계에서 유지할 과거 길이
MAX_HISTORY = max(
    max(RAIN_LAGS),
    max(CUM_RAIN_LAGS),
    max(EXTRA_RAIN_LAGS),
    max(WATER_LAGS),
    max(ROLL_WINDOWS),
)

READ_COLS_PASS1 = [GRID_COL, TIME_COL, TARGET_COL, RAIN_COL]

READ_COLS_PASS2 = [
    GRID_COL,
    TIME_COL,
    TARGET_COL,
    RAIN_COL,
    WATER_COL,
    *CUM_RAIN_COLS,
    *EXTRA_RAIN_FEATURE_COLS,
    *STATIC_FEATURE_COLS,
]


# =========================================================
# 1. 공통 유틸
# =========================================================
def get_sorted_files(rain_dir: str):
    pattern = os.path.join(rain_dir, "final_*.parquet")
    files = sorted(
        glob.glob(pattern),
        key=lambda x: int(os.path.basename(x).replace("final_", "").replace(".parquet", ""))
    )
    if not files:
        raise FileNotFoundError(f"파일 없음: {pattern}")
    return files


def enforce_key_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    groupby / merge / factorize 에 쓰이는 key 컬럼 dtype 강제 통일
    """
    if GRID_COL in df.columns:
        df[GRID_COL] = np.asarray(df[GRID_COL], dtype=np.int64)

    if YEAR_COL in df.columns:
        df[YEAR_COL] = np.asarray(df[YEAR_COL], dtype=np.int64)

    if MONTH_COL in df.columns:
        df[MONTH_COL] = np.asarray(df[MONTH_COL], dtype=np.int64)

    if TIME_COL in df.columns:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")

    return df


def downcast_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    key 컬럼은 int64 유지, 나머지만 downcast
    """
    key_cols = {GRID_COL, YEAR_COL, MONTH_COL}

    for col in df.columns:
        if col in key_cols:
            continue

        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")

    return df


def preprocess_chunk(df: pd.DataFrame, numeric_zero_fill_cols=None) -> pd.DataFrame:
    """
    중요:
    - key 컬럼은 항상 int64
    - 동적 수치 컬럼은 float32
    """
    if numeric_zero_fill_cols is None:
        numeric_zero_fill_cols = []

    # 기본 key 처리
    df[GRID_COL] = pd.to_numeric(df[GRID_COL], errors="coerce")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[GRID_COL, TIME_COL]).copy()

    df[GRID_COL] = np.asarray(df[GRID_COL], dtype=np.int64)
    df[YEAR_COL] = np.asarray(df[TIME_COL].dt.year, dtype=np.int64)
    df[MONTH_COL] = np.asarray(df[TIME_COL].dt.month, dtype=np.int64)

    # 5~10월만 사용
    df = df[df[MONTH_COL].between(5, 10)].copy()

    # 동적 수치 컬럼
    for col in numeric_zero_fill_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.float32)

    # target
    if TARGET_COL in df.columns:
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(np.int8)

    # water
    if WATER_COL in df.columns:
        df[WATER_COL] = pd.to_numeric(df[WATER_COL], errors="coerce").astype(np.float32)

    # static
    for col in STATIC_FEATURE_COLS:
        if col in df.columns:
            if col == "is_river":
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int8)
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)

    # 중복 제거
    df = df.drop_duplicates(subset=[GRID_COL, TIME_COL]).copy()

    # 마지막으로 key 재강제
    df = enforce_key_dtypes(df)
    return df


def add_lag_delta_features(df: pd.DataFrame, base_col: str, lags: list) -> pd.DataFrame:
    df = enforce_key_dtypes(df)

    grouped = df.groupby([YEAR_COL, GRID_COL], sort=False)

    for lag in lags:
        df[f"{base_col}_lag_{lag}"] = grouped[base_col].shift(lag)

    lag1 = grouped[base_col].shift(1)
    lag2 = grouped[base_col].shift(2)

    df[f"{base_col}_diff_1"] = lag1 - lag2
    df[f"{base_col}_pct_change_1"] = (lag1 - lag2) / (lag2.abs() + 1e-6)

    for lag in lags:
        if lag == 1:
            continue
        lagk = grouped[base_col].shift(lag)
        df[f"{base_col}_delta_1_{lag}"] = lag1 - lagk

    return df


def add_rolling_features(df: pd.DataFrame, base_col: str, windows: list) -> pd.DataFrame:
    df = enforce_key_dtypes(df)

    grouped = df.groupby([YEAR_COL, GRID_COL], sort=False)
    shifted = grouped[base_col].shift(1)

    temp = pd.DataFrame({
        YEAR_COL: np.asarray(df[YEAR_COL].values, dtype=np.int64),
        GRID_COL: np.asarray(df[GRID_COL].values, dtype=np.int64),
        "shifted": shifted.values
    })

    temp = enforce_key_dtypes(temp)
    g = temp.groupby([YEAR_COL, GRID_COL], sort=False)["shifted"]

    for w in windows:
        df[f"{base_col}_roll_mean_{w}"] = (
            g.rolling(window=w, min_periods=1)
             .mean()
             .reset_index(level=[0, 1], drop=True)
        )
        df[f"{base_col}_roll_max_{w}"] = (
            g.rolling(window=w, min_periods=1)
             .max()
             .reset_index(level=[0, 1], drop=True)
        )
        df[f"{base_col}_roll_min_{w}"] = (
            g.rolling(window=w, min_periods=1)
             .min()
             .reset_index(level=[0, 1], drop=True)
        )
        df[f"{base_col}_roll_std_{w}"] = (
            g.rolling(window=w, min_periods=2)
             .std()
             .reset_index(level=[0, 1], drop=True)
        )

    return df


def build_tail_dict(df: pd.DataFrame, cols: list, history_len: int):
    """
    (year, grid)별 최근 history_len 행만 저장
    """
    df = enforce_key_dtypes(df)

    tail_dict = {}
    grouped = df.groupby([YEAR_COL, GRID_COL], sort=False)

    for (year, grid), sub in grouped:
        tail_dict[(int(year), int(grid))] = sub[cols].tail(history_len).copy()

    return tail_dict


def concat_tail_and_current(tail_dict, current_df: pd.DataFrame, cols: list):
    """
    이전 파일 tail + 현재 chunk 결합
    """
    parts = []

    current_df = enforce_key_dtypes(current_df)
    current_keys = current_df[[YEAR_COL, GRID_COL]].drop_duplicates().copy()
    current_keys = enforce_key_dtypes(current_keys)

    for _, row in current_keys.iterrows():
        key = (int(row[YEAR_COL]), int(row[GRID_COL]))
        if key in tail_dict:
            parts.append(tail_dict[key])

    parts.append(current_df[cols])
    combined = pd.concat(parts, ignore_index=True)
    combined = enforce_key_dtypes(combined)

    return combined


def build_event_key_df(sampled_event_times: dict) -> pd.DataFrame:
    """
    merge용 (year, time) event key dataframe 생성
    """
    event_years = []
    event_times = []

    for year, times in sampled_event_times.items():
        for t in times:
            event_years.append(np.int64(year))
            event_times.append(pd.Timestamp(t))

    event_key_df = pd.DataFrame({
        YEAR_COL: np.asarray(event_years, dtype=np.int64),
        TIME_COL: pd.to_datetime(event_times)
    })
    event_key_df["_selected_event"] = np.int8(1)
    event_key_df = enforce_key_dtypes(event_key_df)

    return event_key_df


# =========================================================
# 2. PASS 1
# event time 추출
# =========================================================
def pass1_extract_event_times(files):
    pos_times_by_year = defaultdict(set)
    rainy_neg_times_by_year = defaultdict(set)

    for i, fp in enumerate(files):
        print(f"[PASS1] {i+1}/{len(files)} {os.path.basename(fp)}")

        df = pd.read_parquet(fp, columns=READ_COLS_PASS1)
        df = preprocess_chunk(df, numeric_zero_fill_cols=[RAIN_COL])

        if df.empty:
            continue

        df["is_rainy_event"] = (df[RAIN_COL] >= RAIN_THRESHOLD).astype(np.int8)

        event_df = (
            df.groupby([YEAR_COL, TIME_COL], sort=False)
              .agg(
                  event_label=(TARGET_COL, "max"),
                  is_rainy_event=("is_rainy_event", "max")
              )
              .reset_index()
        )

        for year, sub in event_df.groupby(YEAR_COL, sort=False):
            pos_times = sub.loc[sub["event_label"] == 1, TIME_COL]
            neg_times = sub.loc[
                (sub["event_label"] == 0) & (sub["is_rainy_event"] == 1),
                TIME_COL
            ]

            pos_times_by_year[int(year)].update(pos_times.tolist())
            rainy_neg_times_by_year[int(year)].update(neg_times.tolist())

        del df, event_df
        gc.collect()

    sampled_event_times = {}

    print("\n[PASS1] event sampling 결과")
    for year in sorted(pos_times_by_year.keys()):
        pos_list = sorted(pos_times_by_year[year])
        neg_list = sorted(rainy_neg_times_by_year[year])

        n_pos = len(pos_list)
        n_neg_avail = len(neg_list)

        if n_pos == 0:
            print(f"year={year}: positive event 0개 -> skip")
            continue

        if n_neg_avail == 0:
            print(f"year={year}: rainy negative event 0개 -> skip")
            continue

        n_neg_target = n_pos * NEG_POS_RATIO
        sample_n = min(n_neg_target, n_neg_avail)

        neg_series = pd.Series(neg_list)
        sampled_neg = neg_series.sample(n=sample_n, random_state=42).tolist()

        selected = set(pos_list) | set(sampled_neg)
        sampled_event_times[year] = selected

        print(
            f"year={year}: pos_events={n_pos}, "
            f"neg_events={sample_n}, ratio={sample_n / n_pos:.2f}"
        )

    return sampled_event_times


# =========================================================
# 3. PASS 2
# 선택된 event만 feature 생성
# =========================================================
def pass2_build_dataset(files, sampled_event_times):
    result_chunks = []

    base_cols_for_tail = [
        GRID_COL, TIME_COL, YEAR_COL, MONTH_COL,
        TARGET_COL, RAIN_COL, WATER_COL,
        *CUM_RAIN_COLS, *EXTRA_RAIN_FEATURE_COLS,
        *STATIC_FEATURE_COLS,
    ]

    tail_dict = {}
    event_key_df = build_event_key_df(sampled_event_times)

    for i, fp in enumerate(files):
        print(f"[PASS2] {i+1}/{len(files)} {os.path.basename(fp)}")

        current_df = pd.read_parquet(fp, columns=READ_COLS_PASS2)
        current_df = preprocess_chunk(
            current_df,
            numeric_zero_fill_cols=[RAIN_COL, *CUM_RAIN_COLS, *EXTRA_RAIN_FEATURE_COLS]
        )

        if current_df.empty:
            continue

        current_years = set(current_df[YEAR_COL].unique().tolist())
        valid_years = {y for y in current_years if y in sampled_event_times}
        if not valid_years:
            continue

        current_df = current_df[current_df[YEAR_COL].isin(valid_years)].copy()
        current_df = enforce_key_dtypes(current_df)

        if current_df.empty:
            continue

        combined_cols = [c for c in base_cols_for_tail if c in current_df.columns]
        combined_df = concat_tail_and_current(tail_dict, current_df, combined_cols)
        combined_df = combined_df.drop_duplicates(subset=[YEAR_COL, GRID_COL, TIME_COL], keep="last").copy()
        combined_df = enforce_key_dtypes(combined_df)

        # label
        combined_df["y"] = combined_df[TARGET_COL].astype(np.int8)
        combined_df["is_rainy_event"] = (combined_df[RAIN_COL] >= RAIN_THRESHOLD).astype(np.int8)

        # feature 생성
        combined_df = add_lag_delta_features(combined_df, RAIN_COL, RAIN_LAGS)
        combined_df = add_rolling_features(combined_df, RAIN_COL, ROLL_WINDOWS)

        for col in CUM_RAIN_COLS:
            combined_df = add_lag_delta_features(combined_df, col, CUM_RAIN_LAGS)

        for col in EXTRA_RAIN_FEATURE_COLS:
            combined_df = add_lag_delta_features(combined_df, col, EXTRA_RAIN_LAGS)
            combined_df = add_rolling_features(combined_df, col, ROLL_WINDOWS)

        combined_df = add_lag_delta_features(combined_df, WATER_COL, WATER_LAGS)
        combined_df = add_rolling_features(combined_df, WATER_COL, ROLL_WINDOWS)

        # 시간 feature
        combined_df["hour"] = combined_df[TIME_COL].dt.hour.astype(np.int8)
        combined_df["dayofweek"] = combined_df[TIME_COL].dt.dayofweek.astype(np.int8)

        combined_df["hour_sin"] = np.sin(2 * np.pi * combined_df["hour"] / 24).astype(np.float32)
        combined_df["hour_cos"] = np.cos(2 * np.pi * combined_df["hour"] / 24).astype(np.float32)
        combined_df["month_sin"] = np.sin(2 * np.pi * combined_df[MONTH_COL] / 12).astype(np.float32)
        combined_df["month_cos"] = np.cos(2 * np.pi * combined_df[MONTH_COL] / 12).astype(np.float32)

        # 현재 chunk에 속한 row만 남기기
        current_keys = current_df[[YEAR_COL, GRID_COL, TIME_COL]].copy()
        current_keys = enforce_key_dtypes(current_keys)
        current_keys["_in_current"] = np.int8(1)

        combined_df = combined_df.merge(
            current_keys,
            on=[YEAR_COL, GRID_COL, TIME_COL],
            how="left"
        )

        combined_df = combined_df[combined_df["_in_current"] == 1].copy()
        combined_df = combined_df.drop(columns=["_in_current"])

        # 선택된 event만 남기기
        combined_df = enforce_key_dtypes(combined_df)
        combined_df = combined_df.merge(
            event_key_df,
            on=[YEAR_COL, TIME_COL],
            how="left"
        )

        selected_df = combined_df[combined_df["_selected_event"] == 1].copy()
        selected_df = selected_df.drop(columns=["_selected_event"])

        # lag 충분하지 않은 row 제거
        required_cols = [f"{RAIN_COL}_lag_1", f"{WATER_COL}_lag_1"]
        required_cols = [c for c in required_cols if c in selected_df.columns]
        selected_df = selected_df.dropna(subset=required_cols)

        if not selected_df.empty:
            result_chunks.append(selected_df)

        # tail 업데이트
        if tail_dict:
            old_tail_df = pd.concat(tail_dict.values(), ignore_index=True)
            old_tail_df = enforce_key_dtypes(old_tail_df)
            tail_source = pd.concat(
                [old_tail_df, current_df[combined_cols]],
                ignore_index=True
            )
        else:
            tail_source = current_df[combined_cols].copy()

        tail_source = tail_source.drop_duplicates(subset=[YEAR_COL, GRID_COL, TIME_COL], keep="last").copy()
        tail_source = enforce_key_dtypes(tail_source)
        tail_dict = build_tail_dict(tail_source, combined_cols, MAX_HISTORY)

        del current_df, combined_df, selected_df, current_keys, tail_source
        gc.collect()

    if not result_chunks:
        raise ValueError("PASS2 결과가 비어 있음")

    final_df = pd.concat(result_chunks, ignore_index=True)
    final_df = final_df.drop_duplicates(subset=[YEAR_COL, GRID_COL, TIME_COL], keep="last").copy()
    final_df = enforce_key_dtypes(final_df)
    final_df = downcast_df(final_df)

    return final_df


# =========================================================
# 4. 실행
# =========================================================
if __name__ == "__main__":
    files = get_sorted_files(RAIN_DIR)

    print("=== PASS 1: event time 추출 ===")
    sampled_event_times = pass1_extract_event_times(files)

    print("\n=== PASS 2: feature 생성 ===")
    final_df = pass2_build_dataset(files, sampled_event_times)

    # 학습 시 제외할 현재시점 컬럼
    exclude_cols = {
        TARGET_COL,
        "y",
        "is_rainy_event",
        RAIN_COL,
        *CUM_RAIN_COLS,
        *EXTRA_RAIN_FEATURE_COLS,
        WATER_COL,
    }

    feature_cols = [c for c in final_df.columns if c not in exclude_cols]

    print("\n=== 최종 event 비율 확인 ===")
    event_summary = (
        final_df.groupby([YEAR_COL, TIME_COL])["y"]
                .max()
                .reset_index()
                .rename(columns={"y": "event_label"})
    )

    ratio_check = (
        event_summary.groupby(YEAR_COL)["event_label"]
                    .agg(
                        pos_events=lambda s: (s == 1).sum(),
                        neg_events=lambda s: (s == 0).sum()
                    )
                    .reset_index()
    )

    ratio_check["neg_pos_ratio"] = ratio_check["neg_events"] / ratio_check["pos_events"].replace(0, np.nan)
    print(ratio_check)

    print("\nfinal_df shape:", final_df.shape)
    print("feature count:", len(feature_cols))

    final_df.to_parquet(OUTPUT_PATH, index=False)
    print("saved:", OUTPUT_PATH)