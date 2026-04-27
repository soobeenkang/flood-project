"""
도시 침수 위험도 예측 모델 - LightGBM + SMOTE + Optuna
시계열 구조를 고려한 데이터 누수 방지 설계

데이터: final_{year}.parquet (다년도 분할 저장)
컬럼: grid_id, time, rain_1h/3h/6h/12h/24h, rain_intensity, rain_max_3h,
      flood, water_level, mean_elevation, is_river
"""

import glob
import os
from pathlib import Path

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix, f1_score
)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ───────────────────────────────────────────────
# 0. 데이터 로드 (다중 parquet 파일)
# ───────────────────────────────────────────────
def load_data(data_dir: str, pattern: str = "final_*.parquet") -> pd.DataFrame:
    """
    여러 연도의 parquet 파일을 일괄 로드하여 단일 DataFrame으로 결합.

    Parameters
    ----------
    data_dir : str
        parquet 파일들이 위치한 디렉토리 경로
    pattern : str
        파일 매칭 패턴 (기본: final_*.parquet)
    """
    search_path = os.path.join(data_dir, pattern)
    files = sorted(glob.glob(search_path))
    if not files:
        raise FileNotFoundError(f"파일을 찾을 수 없음: {search_path}")

    print(f"발견된 파일 수: {len(files)}")
    for f in files:
        print(f"  - {Path(f).name}")

    # 메모리 효율: 필요한 컬럼만 로드 (parquet의 columnar 장점 활용)
    needed_cols = [
        "grid_id", "time",
        "rain_1h", "rain_3h", "rain_6h", "rain_12h", "rain_24h",
        "rain_intensity", "rain_max_3h",
        "flood", "water_level", "mean_elevation", "is_river",
    ]

    dfs = []
    for f in files:
        df_part = pd.read_parquet(f, columns=needed_cols)
        # 하절기(6~9월) 필터를 파일 단위로 미리 적용 → 메모리 절약
        df_part = df_part[df_part["time"].dt.month.isin([6, 7, 8, 9])]
        dfs.append(df_part)
        print(f"  {Path(f).name}: {len(df_part):,} rows")

    df = pd.concat(dfs, ignore_index=True)
    del dfs

    # 정렬은 전체 결합 후 한 번만 (각 파일별 정렬 후 concat은 grid_id 경계에서 깨짐)
    df = df.sort_values(["grid_id", "time"]).reset_index(drop=True)

    print(f"\n총 데이터: {df.shape}, 메모리: {df.memory_usage(deep=True).sum()/1e6:.1f} MB")
    return df


# ───────────────────────────────────────────────
# 1. 피처 엔지니어링
# ───────────────────────────────────────────────
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["grid_id", "time"]).copy()

    # ── 시즌 그룹 키 생성 (시계열 갭 차단) ───────
    # 6~9월만 남긴 데이터이므로 연도가 바뀌면 새 시즌.
    # (grid_id, year)를 그룹 키로 써서 diff/shift/rolling이
    # 9월 말 ↔ 다음해 6월 초 갭을 넘어 계산되지 않도록 함.
    df["season_year"] = df["time"].dt.year
    group_key = ["grid_id", "season_year"]

    # ── 강수 추세 / 변화율 ──────────────────────
    df["rain_diff_1h"] = df.groupby(group_key)["rain_1h"].diff().fillna(0)
    df["rain_diff_3h"] = df.groupby(group_key)["rain_3h"].diff().fillna(0)

    # 강수 가속도 (2차 차분 → 폭우 시작점 포착)
    df["rain_accel"] = df.groupby(group_key)["rain_diff_1h"].diff().fillna(0)

    # 단기/장기 강수 비율 (단기 집중도)
    df["rain_ratio_1_24"] = df["rain_1h"] / (df["rain_24h"] + 1e-6)
    df["rain_ratio_3_12"] = df["rain_3h"] / (df["rain_12h"] + 1e-6)

    # ── 누적 강수 이동평균 (3-step) ─────────────
    df["rain_1h_ma3"] = (
        df.groupby(group_key)["rain_1h"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

    # ── 강수 강도 구간화 (ordinal) ───────────────
    bins = [-np.inf, 0, 2, 10, 30, np.inf]
    labels = [0, 1, 2, 3, 4]
    df["rain_cat"] = pd.cut(df["rain_intensity"], bins=bins, labels=labels).astype(float)

    # ── 지형 × 강수 교호작용 ─────────────────────
    df["topo_rain_risk"] = (
        (1 / (df["mean_elevation"] + 1)) * df["rain_24h"] * df["is_river"]
    )
    df["elev_rain_12h"] = df["rain_12h"] / (df["mean_elevation"] + 1)

    # ── 시간 피처 ────────────────────────────────
    df["hour"] = df["time"].dt.hour
    df["dayofyear"] = df["time"].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # ── water_level: 누수 위험 처리 ─────────────
    # water_level은 침수 직후 급등 → lag 1step 사용 (현재값 제거)
    # 시즌 경계에서도 shift가 넘어가지 않도록 group_key 적용
    df["water_level_lag1"] = df.groupby(group_key)["water_level"].shift(1).fillna(0)
    df = df.drop(columns=["water_level", "season_year"])  # 보조 키도 제거

    return df


# ───────────────────────────────────────────────
# 2. 피처 / 타겟 정의
# ───────────────────────────────────────────────
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
    "water_level_lag1",
    # 시간
    "hour_sin", "hour_cos", "dayofyear",
]
TARGET_COL = "flood"


# ───────────────────────────────────────────────
# 3. 시계열 분할 (leakage 방지)
# ───────────────────────────────────────────────
def time_based_split(df: pd.DataFrame, test_ratio: float = 0.2):
    """
    grid_id 내부가 아닌 전체 시간 축 기준으로 분할.
    미래 데이터가 학습에 포함되지 않도록 보장.
    """
    cutoff = df["time"].quantile(1 - test_ratio)
    train = df[df["time"] < cutoff].copy()
    test = df[df["time"] >= cutoff].copy()
    print(f"\nTrain 기간: {train['time'].min()} ~ {train['time'].max()}")
    print(f"Test  기간: {test['time'].min()} ~ {test['time'].max()}")
    print(f"Train: {train.shape}, Test: {test.shape}")
    print(f"Train flood ratio: {train[TARGET_COL].mean():.4f}")
    print(f"Test  flood ratio: {test[TARGET_COL].mean():.4f}")
    return train, test


# ───────────────────────────────────────────────
# 4. SMOTE (train set에만 적용)
# ───────────────────────────────────────────────
def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42, verbose: bool = True):
    if verbose:
        print(f"SMOTE 적용 전 - 침수: {y_train.sum():,}, 정상: {(y_train==0).sum():,}")
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    if verbose:
        print(f"SMOTE 적용 후 - 침수: {y_res.sum():,}, 정상: {(y_res==0).sum():,}")
    return X_res, y_res


# ───────────────────────────────────────────────
# 5. Optuna 하이퍼파라미터 튜닝
# ───────────────────────────────────────────────
def run_optuna(X_train, y_train, n_trials: int = 75, cv_folds: int = 5):
    tscv = TimeSeriesSplit(n_splits=cv_folds)

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "average_precision",   # PR-AUC (불균형에 적합)
            "verbosity": -1,
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "n_estimators": 500,
        }

        scores = []
        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            if y_tr.sum() < 5:
                continue
            X_tr_s, y_tr_s = apply_smote(X_tr, y_tr, verbose=False)

            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_tr_s, y_tr_s,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(30, verbose=False),
                           lgb.log_evaluation(-1)],
            )
            proba = model.predict_proba(X_val)[:, 1]
            if y_val.sum() == 0:
                continue
            scores.append(average_precision_score(y_val, proba))

        return np.mean(scores) if scores else 0.0

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest PR-AUC (CV): {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study.best_params


# ───────────────────────────────────────────────
# 6. 최종 모델 학습 및 평가
# ───────────────────────────────────────────────
def train_final_model(best_params: dict, X_train, y_train, X_test, y_test):
    params = {
        "objective": "binary",
        "metric": "average_precision",
        "verbosity": -1,
        "n_estimators": 1000,
        **best_params,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(100)],
    )
    return model


def evaluate(model, X_test, y_test, threshold: float = 0.3):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)

    print("\n" + "="*50)
    print(f"평가 임계값: {threshold}")
    print("="*50)
    print(classification_report(y_test, pred, target_names=["정상", "침수"]))
    print(f"ROC-AUC  : {roc_auc_score(y_test, proba):.4f}")
    print(f"PR-AUC   : {average_precision_score(y_test, proba):.4f}")
    print(f"F1 (침수): {f1_score(y_test, pred):.4f}")
    print("\n혼동행렬:")
    print(confusion_matrix(y_test, pred))

    # 최적 임계값 탐색 (F1 기준)
    precision, recall, thresholds = precision_recall_curve(y_test, proba)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_thresh = thresholds[np.argmax(f1_scores[:-1])]
    print(f"\n최적 임계값 (F1 최대): {best_thresh:.3f}")
    return proba, best_thresh


def feature_importance_report(model, feature_cols):
    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\n── Top 15 피처 중요도 ──")
    print(imp.head(15).to_string(index=False))
    return imp


# ───────────────────────────────────────────────
# 7. 메인 파이프라인
# ───────────────────────────────────────────────
def main(data_dir: str, n_trials: int = 75, pattern: str = "final_*.parquet"):
    print("=" * 50)
    print("  도시 침수 예측 - LightGBM 파이프라인")
    print("=" * 50)

    # 1. 데이터 로드 (다중 parquet 결합)
    df = load_data(data_dir, pattern=pattern)
    print(f"\n전체 침수 비율: {df[TARGET_COL].mean():.4f}")

    # 2. 피처 엔지니어링
    df = feature_engineering(df)

    # 3. 사용 가능한 피처만 선택
    available = [c for c in FEATURE_COLS if c in df.columns]
    print(f"\n사용 피처 ({len(available)}개): {available}")

    # 4. 시계열 분할
    train_df, test_df = time_based_split(df)
    X_train = train_df[available].reset_index(drop=True)
    y_train = train_df[TARGET_COL].astype(int).reset_index(drop=True)
    X_test  = test_df[available].reset_index(drop=True)
    y_test  = test_df[TARGET_COL].astype(int).reset_index(drop=True)

    # 메모리 정리
    del df, train_df, test_df

    # 5. Optuna 튜닝
    print(f"\nOptuna 튜닝 시작 ({n_trials} trials)...")
    best_params = run_optuna(X_train, y_train, n_trials=n_trials)

    # 6. 전체 train에 SMOTE 후 최종 학습
    print("\n최종 모델 학습용 SMOTE 적용 중...")
    X_tr_s, y_tr_s = apply_smote(X_train, y_train)
    model = train_final_model(best_params, X_tr_s, y_tr_s, X_test, y_test)

    # 7. 평가
    proba, best_thresh = evaluate(model, X_test, y_test, threshold=0.3)

    # 8. 피처 중요도
    feature_importance_report(model, available)

    # 9. 모델 저장
    model.booster_.save_model("flood_lgbm_model.txt")
    print("\n모델 저장 완료: flood_lgbm_model.txt")

    return model, best_thresh, proba


if __name__ == "__main__":
    # ↓ parquet 파일들이 들어있는 디렉토리 경로 지정
    model, threshold, probabilities = main(
        data_dir="./data",          # final_2010.parquet, final_2011.parquet, ... 들어있는 폴더
        pattern="final_*.parquet",  # 매칭 패턴
        n_trials=75,
    )