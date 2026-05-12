"""
build_dataset.py

lgbm 학습용 데이터 빌더
-> 데이터 사이즈 확인하고 ratio, threshold 개선요

입력:
  - 침수흔적도:  grid_id, F_SAT_YMD, F_END_YVM, IS_FLOODED
  - 고도:        grid_id, mean_elevation
  - 하천:        grid_id, is_river
  - aws 강수 (연도별): grid_id, time, rain_1h, rain_3h, rain_6h, rain_12h,
                              rain_24h, rain_intensity, rain_max_3h

출력 (연도별 parquet):
  final_YYYY.parquet
  cols = [grid_id, time, rain_1h..rain_max_3h,
          mean_elevation, is_river, is_flooded]

샘플링 기준:
  - 1 : 어떤 grid라도 침수 기간에 해당하는 time
    → 그 시간의 모든 grid 데이터 저장
  - 2 - neg case : 어디도 침수가 아닐때, 
      어떤 grid라도 rain > NEG_TIME_RAIN_THRESHOLD 인 time
"""

import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd


# ───────────────────────────────────────────────
# 경로/설정 - 수정 필요
# ───────────────────────────────────────────────
FLOOD_PATH        = "data/flood_history.parquet"
ELEVATION_PATH    = "data/elevation.parquet"
RIVER_PATH        = "data/river.parquet"
RAIN_DIR          = "data/rain"
RAIN_PREFIX       = "rain_"

OUT_DIR           = "data/final"
OUT_PREFIX        = "final_"

# neg case 샘플링 비율 (1.0 = 강수 > NEG_TIME_RAIN_THRESHOLD 시간 전부 사용)
NEG_TIME_SAMPLE_RATIO = 1.0
# neg case 정의 시 어떤 grid라도 이 강수 이상이어야 함
NEG_TIME_RAIN_THRESHOLD = 5.0
# neg case 샘플링용 시드
RANDOM_STATE = 42


# ───────────────────────────────────────────────
# 1. 정적 grid 피처 로드 (고도 + 하천)
# ───────────────────────────────────────────────
def load_static_grid_features():
    elev = pd.read_parquet(ELEVATION_PATH)[["grid_id", "mean_elevation"]]
    river = pd.read_parquet(RIVER_PATH)[["grid_id", "is_river"]]
    elev["grid_id"] = elev["grid_id"].astype("int32")
    river["grid_id"] = river["grid_id"].astype("int32")
    static = elev.merge(river, on="grid_id", how="outer")
    print(f"[static] grids: {len(static):,} "
          f"(elev {len(elev):,}, river {len(river):,})")
    return static


# ───────────────────────────────────────────────
# 2. 침수흔적도 → (grid_id, hour) 양성 집합
#    한 행이 침수 기간이므로 시간(hour) 단위로 펼친다.
# ───────────────────────────────────────────────
def expand_flood_to_hourly(flood_df):
    """
    flood_df: grid_id, F_SAT_YMD, F_END_YVM, IS_FLOODED
    return:   DataFrame(grid_id, flood_time)  ← (grid, hour) 양성 집합
              flood_time은 정시 (예: 2018-08-28 15:00:00)
    """
    df = flood_df.copy()
    df = df[df["IS_FLOODED"] == 1]
    df["F_SAT_YMD"] = pd.to_datetime(df["F_SAT_YMD"])
    df["F_END_YVM"] = pd.to_datetime(df["F_END_YVM"])

    df["start_h"] = df["F_SAT_YMD"].dt.ceil("h")
    df["end_h"]   = df["F_END_YVM"].dt.floor("h")

    # 유효한 기간만 (start <= end)
    df = df[df["start_h"] <= df["end_h"]].copy()

    # 각 행마다 정시 시퀀스 생성 후 explode
    df["flood_time"] = df.apply(
        lambda r: pd.date_range(r["start_h"], r["end_h"], freq="h"),
        axis=1,
    )
    out = df[["grid_id", "flood_time"]].explode("flood_time", ignore_index=True)
    out["grid_id"] = out["grid_id"].astype("int32")
    out["flood_time"] = pd.to_datetime(out["flood_time"])
    out = out.drop_duplicates().reset_index(drop=True)
    print(f"[flood] (grid, hour) 양성 pair: {len(out):,}")
    return out


def build_year(year, rain_path, static_df, flood_hourly, out_path):
    print(f"\n========== {year} ==========")

    # 3-1. 강수 로드
    rain = pd.read_parquet(rain_path)
    rain["grid_id"] = rain["grid_id"].astype("int32")
    rain["time"] = pd.to_datetime(rain["time"])
    print(f"  rain rows: {len(rain):,}")

    # 3-2. 해당 연도의 양성 (grid, hour) 만 필터
    yr_flood = flood_hourly[
        flood_hourly["flood_time"].dt.year == year
    ].copy()
    print(f"  flood (grid, hour) in {year}: {len(yr_flood):,}")

    # 3-3. rain의 time과 직접 매칭
    yr_flood = yr_flood.rename(columns={"flood_time": "time"})
    yr_flood["is_flooded"] = np.int8(1)

    merged = rain.merge(
        yr_flood[["grid_id", "time", "is_flooded"]],
        on=["grid_id", "time"],
        how="left",
    )
    merged["is_flooded"] = merged["is_flooded"].fillna(0).astype("int8")

    n_pos_rows = int(merged["is_flooded"].sum())
    print(f"  is_flooded=1 rows (raw): {n_pos_rows:,}")

    # 3-4. is_flooded=1 row가 하나라도 있는 time 필터
    pos_times = merged.loc[merged["is_flooded"] == 1, "time"].unique()
    pos_times_set = set(pos_times)
    print(f"  positive timestamps: {len(pos_times):,}")

    # 3-5. 나머지 중 어떤 grid라도 rain > threshold 필터
    is_neg_time_candidate = ~merged["time"].isin(pos_times_set)

    rain_time_flag = (
        merged.loc[is_neg_time_candidate]
        .groupby("time")["rain_1h"]
        .max()
    )
    neg_times_all = rain_time_flag[rain_time_flag > NEG_TIME_RAIN_THRESHOLD].index
    print(f"  negative candidate timestamps (rain>{NEG_TIME_RAIN_THRESHOLD}): "
          f"{len(neg_times_all):,}")

    # 3-6. neg case 샘플링
    if NEG_TIME_SAMPLE_RATIO < 1.0:
        rng = np.random.default_rng(RANDOM_STATE + year)
        n_sample = int(len(neg_times_all) * NEG_TIME_SAMPLE_RATIO)
        neg_times = rng.choice(neg_times_all.values, size=n_sample, replace=False)
        neg_times = pd.DatetimeIndex(neg_times)
    else:
        neg_times = neg_times_all
    print(f"  negative timestamps sampled: {len(neg_times):,}")

    # 3-7. 최종 row 선택: 1 + 2
    keep_times = pd.DatetimeIndex(pos_times).union(pd.DatetimeIndex(neg_times))
    out = merged[merged["time"].isin(keep_times)].copy()
    del merged

    # 3-8. 정적 피처 머지
    out = out.merge(static_df, on="grid_id", how="left")

    # 3-9. 컬럼 정리
    feature_cols = [
        "grid_id", "time",
        "rain_1h", "rain_3h", "rain_6h", "rain_12h",
        "rain_24h", "rain_intensity", "rain_max_3h",
        "mean_elevation", "is_river",
        "is_flooded",
    ]
    out = out[feature_cols]

    for c in ["rain_1h", "rain_3h", "rain_6h", "rain_12h",
              "rain_24h", "rain_intensity", "rain_max_3h",
              "mean_elevation"]:
        out[c] = out[c].astype("float32")
    out["is_river"] = out["is_river"].astype("int8")
    out["is_flooded"] = out["is_flooded"].astype("int8")
    out["grid_id"] = out["grid_id"].astype("int32")

    n_pos = int((out["is_flooded"] == 1).sum())
    n_neg = len(out) - n_pos
    mem_mb = out.memory_usage(deep=True).sum() / 1e6
    print(f"  final: pos {n_pos:,} + neg {n_neg:,} = {len(out):,} ({mem_mb:.0f}MB)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_parquet(out_path, index=False, compression="snappy")
    print(f"  saved: {out_path}")

    del out


# ───────────────────────────────────────────────
# main
# ───────────────────────────────────────────────
def main():
    # 정적 피처
    static_df = load_static_grid_features()

    # 침수흔적도 일자별 펼치기
    flood_raw = pd.read_parquet(FLOOD_PATH)
    flood_hourly = expand_flood_to_hourly(flood_raw)
    del flood_raw

    # 연도별 강수 파일 발견
    rain_files = sorted(glob.glob(os.path.join(RAIN_DIR, f"{RAIN_PREFIX}*.parquet")))
    print(f"\nrain files: {len(rain_files)}")
    for f in rain_files:
        print(f"  {f}")

    for f in rain_files:
        year = int(Path(f).stem.replace(RAIN_PREFIX, ""))
        out_path = os.path.join(OUT_DIR, f"{OUT_PREFIX}{year}.parquet")
        if os.path.exists(out_path):
            print(f"\n[skip] {out_path} already exists")
            continue
        build_year(year, f, static_df, flood_hourly, out_path)


if __name__ == "__main__":
    main()