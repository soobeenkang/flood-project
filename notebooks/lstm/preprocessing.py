# =========================
# LSTM 학습용 데이터셋 생성 코드
# =========================
# 이 파일의 목적:
# raw 데이터 → feature 생성 → 시퀀스 생성 → PyTorch Dataset까지 만드는 것

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# =========================
# 설정
# =========================

@dataclass
class SequenceConfig:
    """
    LSTM 데이터셋 생성 전체 설정

    ✔ 여기서 바꾸면 전체 데이터 파이프라인이 바뀜
    """

    # ===== 데이터 경로 =====
    data_path: str = "data/output/flood_model_input.parquet"

    # ===== 기본 컬럼 =====
    time_col: str = "time"          # 시간 컬럼
    group_col: str = "grid_id"      # 시계열 그룹 (격자 / 센서 단위)
    target_col: str = "flooded"     # 예측 대상 (0: 정상, 1: 침수)

    # ===== 입력 feature =====
    rain_col: str = "rainfall"          # 강수량

    # ===== static =====
    elevation_col: str = "elevation"    # 고도 (static)
    river_col: str = "is_river"         # 강 근접 여부 (static)

    # ===== 추가 feature =====
    extra_dynamic_cols: Optional[List[str]] = None
    static_cols: Optional[List[str]] = None

    # ===== 시퀀스 설정 =====
    seq_len: int = 12
    """
    LSTM 입력 길이
    → 예: 12이면 최근 12개 시점 사용
    """

    pred_horizon: int = 1
    """
    예측 시점
    0 → 현재 시점
    1 → 다음 시점
    """

    # ===== 시간 기반 분할 =====
    n_chunks: int = 5
    test_chunk_ids: Tuple[int, ...] = (4,)
    """
    전체 데이터를 시간순으로 나눔
    → 마지막 chunk를 test로 사용
    """

    # ===== feature 옵션 =====
    make_rain_delta: bool = True
    make_rain_rolling_min: bool = True
    make_time_flags: bool = True

    # ===== 기타 =====
    fillna_value: float = 0.0
    enforce_time_sort: bool = True


# =========================
# 데이터 로딩
# =========================

def load_dataframe(path: str) -> pd.DataFrame:
    """파일을 읽어서 DataFrame으로 변환"""
    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"파일 없음: {path}")

    if path_obj.suffix == ".parquet":
        return pd.read_parquet(path_obj)
    elif path_obj.suffix == ".csv":
        return pd.read_csv(path_obj)
    else:
        raise ValueError("지원하지 않는 파일 형식")


def ensure_datetime(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """time 컬럼을 datetime으로 변환"""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    return df


def reduce_columns(df: pd.DataFrame, keep_cols: List[str]) -> pd.DataFrame:
    """필요한 컬럼만 남김"""
    existing = [c for c in keep_cols if c in df.columns]
    return df[existing].copy()


# =========================
# Feature Engineering
# =========================

def add_rain_features(df: pd.DataFrame, config: SequenceConfig) -> pd.DataFrame:
    """
    강수 기반 feature 생성

    ✔ 핵심:
    - 누적 강수량
    - 변화량
    """

    df = df.copy()
    df = df.sort_values([config.group_col, config.time_col])

    grouped = df.groupby(config.group_col)

    # 누적 강수량
    df["rain_3h_sum"] = grouped[config.rain_col].transform(lambda s: s.rolling(3, 1).sum())
    df["rain_12h_sum"] = grouped[config.rain_col].transform(lambda s: s.rolling(12, 1).sum())
    df["rain_24h_sum"] = grouped[config.rain_col].transform(lambda s: s.rolling(24, 1).sum())

    # 평균
    df["rain_3h_mean"] = grouped[config.rain_col].transform(lambda s: s.rolling(3, 1).mean())

    # 최소값
    if config.make_rain_rolling_min:
        df["rain_3h_min"] = grouped[config.rain_col].transform(lambda s: s.rolling(3, 1).min())

    # 변화량
    if config.make_rain_delta:
        df["rain_diff_1"] = grouped[config.rain_col].diff(1)
        df["rain_diff_3"] = grouped[config.rain_col].diff(3)

    return df


def add_time_flags(df: pd.DataFrame, config: SequenceConfig) -> pd.DataFrame:
    """
    시간 기반 feature

    ✔ 계절/시간 패턴 학습 가능하게 함
    """
    df = df.copy()
    t = df[config.time_col]

    df["month"] = t.dt.month
    df["hour"] = t.dt.hour
    df["weekday"] = t.dt.weekday

    # 장마 시즌
    df["is_monsoon"] = t.dt.month.isin([6, 7, 8, 9]).astype(int)

    return df


def build_features(df, config):
    """
    전체 feature 생성 + dynamic/static 분리
    """

    df = df.copy()

    required_cols = [
        config.time_col,
        config.group_col,
        config.target_col,
        config.rain_col,
        config.elevation_col,
        config.river_col,
    ]

    df = reduce_columns(df, required_cols)
    df = ensure_datetime(df, config.time_col)

    df = add_rain_features(df, config)

    if config.make_time_flags:
        df = add_time_flags(df, config)

    # static feature
    static_features = [config.elevation_col, config.river_col]

    # dynamic feature
    dynamic_features = [
        config.rain_col,
        "rain_3h_sum",
        "rain_12h_sum",
        "rain_24h_sum",
        "rain_3h_mean",
    ]

    if config.make_rain_delta:
        dynamic_features += ["rain_diff_1", "rain_diff_3"]
        
# =========================
# chunk 파일 저장
# =========================

def save_chunks(df, save_dir="data/chunks"):
    from pathlib import Path
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for chunk_id, g in df.groupby("chunk_id"):
        path = f"{save_dir}/chunk_{chunk_id}.parquet"
        g.to_parquet(path)
        print(f"saved: {path}")

    if config.make_time_flags:
        dynamic_features += ["month", "hour", "weekday", "is_monsoon"]

    # 결측치 처리
    df[dynamic_features] = df[dynamic_features].fillna(config.fillna_value)
    df[static_features] = df[static_features].fillna(config.fillna_value)

    # target 정리
    df[config.target_col] = df[config.target_col].fillna(0).astype(int)

    return df, dynamic_features, static_features


# =========================
# 시간 기반 chunk 분할
# =========================

def assign_time_chunks(df, config):
    """
    데이터를 시간 기준으로 나눔 (랜덤 X)

    ✔ 이유:
    미래 데이터가 train에 들어가는 것 방지
    """

    unique_times = sorted(df[config.time_col].unique())
    splits = np.array_split(unique_times, config.n_chunks)

    mapping = {}
    for i, chunk in enumerate(splits):
        for t in chunk:
            mapping[t] = i

    df["chunk_id"] = df[config.time_col].map(mapping)
    return df


# =========================
# 시퀀스 생성 (핵심)
# =========================

def build_sequences_from_group(group_df, config, dyn_feats, static_feats):
    """
    하나의 grid에서 sliding window 생성

    ✔ 출력:
    X_seq   : [seq_len, dynamic_dim]
    X_static: [static_dim]
    y       : 0 or 1
    """

    seq_len = config.seq_len
    horizon = config.pred_horizon

    dyn = group_df[dyn_feats].values
    stat = group_df[static_feats].values
    y_all = group_df[config.target_col].values

    results = []

    for i in range(len(group_df) - seq_len - horizon + 1):
        x_seq = dyn[i:i+seq_len]
        x_static = stat[i+seq_len-1]
        y = y_all[i+seq_len-1 + horizon]

        results.append((x_seq, x_static, y))

    return results


def build_all_sequences(df, config, dyn_feats, static_feats):
    """전체 그룹에 대해 시퀀스 생성"""
    data = []

    for gid, g in df.groupby(config.group_col):
        seqs = build_sequences_from_group(g, config, dyn_feats, static_feats)

        for x_seq, x_static, y in seqs:
            data.append({
                "X_seq": x_seq,
                "X_static": x_static,
                "y": y,
                "group_id": gid,
                "chunk_id": g["chunk_id"].iloc[-1],
            })

    return pd.DataFrame(data)


# =========================
# train / test 분리
# =========================

def split_train_test(seq_df, config):
    test_mask = seq_df["chunk_id"].isin(config.test_chunk_ids)
    return seq_df[~test_mask], seq_df[test_mask]


# =========================
# PyTorch Dataset
# =========================

class FloodDataset(Dataset):
    """PyTorch용 Dataset"""

    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        return {
            "x_seq": torch.tensor(row["X_seq"], dtype=torch.float32),
            "x_static": torch.tensor(row["X_static"], dtype=torch.float32),
            "y": torch.tensor(row["y"], dtype=torch.float32),
        }

# =========================
# chunk 파일 저장
# =========================

def save_chunks(df, save_dir="data/chunks"):
    from pathlib import Path
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for chunk_id, g in df.groupby("chunk_id"):
        path = f"{save_dir}/chunk_{chunk_id}.parquet"
        g.to_parquet(path)
        print(f"saved: {path}")



# =========================
# 전체 파이프라인
# =========================

def make_lstm_datasets(config):
    df = load_dataframe(config.data_path)

    df, dyn, stat = build_features(df, config)
    df = assign_time_chunks(df, config)

    save_chunks(df)

    seq_df = build_all_sequences(df, config, dyn, stat)

    train_df, test_df = split_train_test(seq_df, config)

    return FloodDataset(train_df), FloodDataset(test_df)




# =========================
# 실행
# =========================

if __name__ == "__main__":
    config = SequenceConfig()

    train_ds, test_ds = make_lstm_datasets(config)

    sample = train_ds[0]

    print(sample["x_seq"].shape)     # [seq_len, feature]
    print(sample["x_static"].shape)
    print(sample["y"])