"""
피처 엔지니어링 1회성 빌드 스크립트.

원본 final_YYYY.parquet → 피처 엔지니어링 → cache_features/feat_YYYY.parquet 저장.

언제 실행?
- 처음 한 번
- 피처 정의(engineer_one_file 함수)를 바꿨을 때
- 원본 데이터가 바뀌었을 때

학습/하이퍼파라미터만 바꾼다면 이 스크립트는 다시 실행할 필요 없음.

사용:
    python build_features.py              # 캐시 있으면 스킵, 없는 연도만 빌드
    python build_features.py --force      # 전체 재생성
"""

import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    RAW_DATA_DIR, RAW_PATTERN, CACHE_DIR, CACHE_PREFIX, RAW_COLS,
)


# ───────────────────────────────────────────────
# 유틸
# ───────────────────────────────────────────────
def downcast_df(df: pd.DataFrame) -> pd.DataFrame:
    """메모리 절감용 dtype 다운캐스트."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def _extract_year(filename: str) -> int:
    m = re.search(r"(\d{4})", Path(filename).stem)
    if not m:
        raise ValueError(f"파일명에서 연도 추출 실패: {filename}")
    return int(m.group(1))


# ───────────────────────────────────────────────
# 핵심 피처 엔지니어링 (단일 파일/단일 연도)
# ───────────────────────────────────────────────
def engineer_one_file(df: pd.DataFrame) -> pd.DataFrame:
    """
    한 연도 데이터에 대해 피처 엔지니어링.
    파일이 단일 연도 + 6~9월이므로 grid_id만으로 group해도 시즌 갭 안전.

    피처를 추가/수정하려면 이 함수만 수정하고 build_features.py를 재실행하면 됨.
    """
    df = df.sort_values(["grid_id", "time"]).copy()
    g = df.groupby("grid_id", observed=True, sort=False)

    # 강수 추세 / 변화율
    df["rain_diff_1h"] = g["rain_1h"].diff().fillna(0)
    df["rain_diff_3h"] = g["rain_3h"].diff().fillna(0)
    df["rain_accel"] = (
        df.groupby("grid_id", observed=True, sort=False)["rain_diff_1h"]
        .diff()
        .fillna(0)
    )

    # 단기/장기 강수 비율
    df["rain_ratio_1_24"] = df["rain_1h"] / (df["rain_24h"] + 1e-6)
    df["rain_ratio_3_12"] = df["rain_3h"] / (df["rain_12h"] + 1e-6)

    # 누적 강수 이동평균 (3-step)
    df["rain_1h_ma3"] = g["rain_1h"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

    # 강수 강도 구간화
    bins = [-np.inf, 0, 2, 10, 30, np.inf]
    df["rain_cat"] = pd.cut(
        df["rain_intensity"], bins=bins, labels=[0, 1, 2, 3, 4]
    ).astype("float32")

    # 지형 × 강수 교호작용
    df["topo_rain_risk"] = (
        (1 / (df["mean_elevation"] + 1)) * df["rain_24h"] * df["is_river"]
    )
    df["elev_rain_12h"] = df["rain_12h"] / (df["mean_elevation"] + 1)

    # 시간 피처: datetime → 정수 분해 (numpy 호환성 확보)
    df["year"] = df["time"].dt.year.astype("int16")
    df["month"] = df["time"].dt.month.astype("int8")
    df["day"] = df["time"].dt.day.astype("int8")
    df["hour"] = df["time"].dt.hour.astype("int8")
    df["dayofyear"] = df["time"].dt.dayofyear.astype("int16")
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).astype("float32")

    # water_level: 누수 차단을 위해 lag 1step만 사용
    df["water_level_lag1"] = (
        g["water_level"].shift(1).fillna(0).astype("float32")
    )
    df = df.drop(columns=["water_level"])

    df = downcast_df(df)
    return df


# ───────────────────────────────────────────────
# 빌드 파이프라인
# ───────────────────────────────────────────────
def build_all(force_rebuild: bool = False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, RAW_PATTERN)))
    if not files:
        raise FileNotFoundError(
            f"원본 파일 없음: {RAW_DATA_DIR}/{RAW_PATTERN}"
        )

    print(f"발견된 원본 파일 수: {len(files)}")
    built, skipped = 0, 0

    for f in files:
        year = _extract_year(f)
        cache_path = os.path.join(CACHE_DIR, f"{CACHE_PREFIX}{year}.parquet")

        if os.path.exists(cache_path) and not force_rebuild:
            print(f"  [skip] {year} (캐시 존재)")
            skipped += 1
            continue

        print(f"  [build] {year} ...", end=" ", flush=True)
        df = pd.read_parquet(f, columns=RAW_COLS)

        # 6~9월만
        df = df[df["time"].dt.month.isin([6, 7, 8, 9])]

        # grid_id를 category로 압축 (메모리 절감 + groupby 가속)
        df["grid_id"] = df["grid_id"].astype("category")
        df = downcast_df(df)

        df = engineer_one_file(df)

        df.to_parquet(cache_path, compression="snappy", index=False)
        mem_mb = df.memory_usage(deep=True).sum() / 1e6
        print(f"rows={len(df):,}, mem={mem_mb:.0f}MB → {Path(cache_path).name}")
        del df
        built += 1

    print(f"\n완료 - 빌드: {built}, 스킵: {skipped}")
    print(f"캐시 위치: {CACHE_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force", action="store_true",
        help="캐시 무시하고 전체 재생성"
    )
    args = parser.parse_args()
    build_all(force_rebuild=args.force)