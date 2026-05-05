import glob
import os
from pathlib import Path
import json

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna

from optuna.samplers import TPESampler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from config import (
    CACHE_DIR, CACHE_PREFIX,
    FEATURE_COLS, TARGET_COL, LOAD_COLS,
    TEST_YEARS, N_TRIALS, MODEL_PATH,
)


# ───────────────────────────────────────────────
# 캐시 로드
# ───────────────────────────────────────────────
def discover_cached_years():
    files = sorted(glob.glob(os.path.join(CACHE_DIR, f"{CACHE_PREFIX}*.parquet")))
    return {
        int(Path(f).stem.replace(CACHE_PREFIX, "")): f
        for f in files
    }


# ───────────────────────────────────────────────
# 메모리 로드 + 필터링 - 각 파일 한번씩 메모리에 올리며
# 양성은 누적, 음성은 양성 기준 hard neg만 누적
# 마지막에 음성 언더 샘플링
# ───────────────────────────────────────────────
def load_and_filter_streaming(year_map, train_years,
                              pos_quantile=0.05,
                              neg_pos_ratio=20,
                              random_state=42):
    print("\n[stream] step 1: collec pos + calculate rain")

    pos_list = []
    for y in train_years:
        df = pd.read_parquet(year_map[y], columns=LOAD_COLS)
        pos_list.append(df[df[TARGET_COL] == 1].copy())
        n_pos_y = (df[TARGET_COL] == 1).sum()
        print(f"  {y}: positive {n_pos_y:,}")
        del df

    all_pos = pd.concat(pos_list, ignore_index=True)
    del pos_list
    print(f"  → all positive: {len(all_pos):,}")

    # ── 추가: rain_24h = 0인 양성 제거 (강수 외 원인 침수 노이즈) ──
    n_before = len(all_pos)
    all_pos = all_pos[all_pos["rain_24h"] > 0].reset_index(drop=True)
    print(f"  rain_24h=0 양성 제외: {n_before:,} → {len(all_pos):,} "
        f"({(n_before-len(all_pos))/n_before*100:.1f}% 제거)")
    
    """
    분포 확인 위한 코드
    print("\n[진단] 양성의 강수 분포:")
    for col in ["rain_1h", "rain_3h", "rain_6h", "rain_12h", "rain_24h"]:
        s = all_pos[col]
        zero_pct = (s == 0).mean() * 100
        nz = s[s > 0]  # 0이 아닌 값만
        print(f"  {col:>9s}: zero={zero_pct:5.1f}%, "
            f"median(non-zero)={nz.median() if len(nz) else 0:.2f}, "
            f"q05={s.quantile(0.05):.2f}, q50={s.quantile(0.5):.2f}")
    """

    # 양성 강수 분포 기준 임계값
    threshold_24h = all_pos["rain_24h"].quantile(pos_quantile)
    print(f"\n  positive rain {pos_quantile*100:.0f}% :")
    print(f"    rain_24h ≥ {threshold_24h:.2f}mm")

    # 2단계: 각 연도에서 hard negative만 뽑기
    print("\n[STREAM] step 2: collect yearly hard negative")
    neg_list = []
    total_neg_raw = 0
    for y in train_years:
        df = pd.read_parquet(year_map[y], columns=LOAD_COLS)
        neg = df[df[TARGET_COL] == 0]
        total_neg_raw += len(neg)
        hard_neg = neg[(neg["rain_24h"] >= threshold_24h)].copy()
        print(f"  {y}: negative {len(neg):,} → hard {len(hard_neg):,}")
        neg_list.append(hard_neg)
        del df, neg, hard_neg

    all_neg = pd.concat(neg_list, ignore_index=True)
    del neg_list
    print(f"  → negative {total_neg_raw:,} → hard {len(all_neg):,} "
          f"({len(all_neg)/total_neg_raw*100:.1f}%)")

    # 3단계: 음성 언더샘플링
    target_n = len(all_pos) * neg_pos_ratio
    if len(all_neg) > target_n:
        all_neg = all_neg.sample(n=target_n, random_state=random_state)
        print(f"\n  undersampling: → {len(all_neg):,} (pos×{neg_pos_ratio})")

    # 4단계: 합치기
    out = pd.concat([all_pos, all_neg], ignore_index=True)
    del all_pos, all_neg
    out = out.sample(frac=1, random_state=random_state).reset_index(drop=True)

    n_pos = int((out[TARGET_COL] == 1).sum())
    n_neg = len(out) - n_pos
    mem_mb = out.memory_usage(deep=True).sum() / 1e6
    print(f"\n  final: pos {n_pos:,} + neg {n_neg:,} = total {len(out):,} "
          f"({mem_mb:.0f}MB)")

    return out


# ───────────────────────────────────────────────
# Optuna용 샘플
# ───────────────────────────────────────────────
def stratified_sample(df, n_total=500_000, random_state=42):
    if len(df) <= n_total:
        return df
    
    pos = df[df[TARGET_COL] == 1]
    neg = df[df[TARGET_COL] == 0]
    pos_ratio = len(pos) / len(df)
    n_pos = int(n_total * pos_ratio)
    n_neg = n_total - n_pos
    pos_s = pos.sample(n=min(len(pos), n_pos), random_state=random_state)
    neg_s = neg.sample(n=min(len(neg), n_neg), random_state=random_state)
    out = pd.concat([pos_s, neg_s])
    return out.sample(frac=1, random_state=random_state).reset_index(drop=True)


# ───────────────────────────────────────────────
# Optuna
# ───────────────────────────────────────────────
def run_optuna(sample_df, scale_pos_weight):
    X = sample_df[FEATURE_COLS]
    y = sample_df[TARGET_COL]
    tscv = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "average_precision",
            "verbosity": -1,
            "scale_pos_weight": scale_pos_weight,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 256),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 500),

            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),

            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),

            "n_estimators": 500,
        }

        scores = []

        for tr_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
            if y_tr.sum() < 5 or y_val.sum() == 0:
                continue
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(30, verbose=False),
                           lgb.log_evaluation(-1)],
            )

            proba = model.predict_proba(X_val)[:, 1]
            scores.append(average_precision_score(y_val, proba))

        return np.mean(scores) if scores else 0.0

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    print(f"\nbest cv pr=auc: {study.best_value:.4f}")
    print(f"best params: {study.best_params}")

    return study.best_params


# ───────────────────────────────────────────────
# threshold (recall 중심)
# ───────────────────────────────────────────────
def find_threshold_for_recall(y_true, proba, target_recall=0.6):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)

    p, r = precision[:-1], recall[:-1]
    mask = r >= target_recall
    if not mask.any():
        return 0.5
    
    best = np.where(mask)[0][np.argmax(p[mask])]
    
    return float(thresholds[best])


# ───────────────────────────────────────────────
# streaming test (핵심)
# ───────────────────────────────────────────────
def evaluate_streaming(model, year_map, test_years):
    all_p, all_y = [], []

    for y in test_years:
        print(f"[TEST] {y}")
        df = pd.read_parquet(year_map[y], columns=LOAD_COLS)

        X = df[FEATURE_COLS]
        y_true = df[TARGET_COL].astype("int8")

        proba = model.predict_proba(X)[:, 1]

        all_p.append(proba)
        all_y.append(y_true.values)

        del df, X, y_true

    return np.concatenate(all_y), np.concatenate(all_p)


# ───────────────────────────────────────────────
# main
# ───────────────────────────────────────────────
def main():
    year_map = discover_cached_years()
    years = sorted(year_map.keys())

    train_years = years[:-TEST_YEARS]
    test_years = years[-TEST_YEARS:]

    print("Train:", train_years)
    print("Test :", test_years)

    # ─────────────────────────────────────────
    # 1. load + filtering for 16GB
    # ─────────────────────────────────────────
    train_df = load_and_filter_streaming(
        year_map, train_years,
        pos_quantile=0.05,
        neg_pos_ratio=20,
    )

    # ─────────────────────────────────────────
    # 2.  check pos vs neg ratio
    # ─────────────────────────────────────────
    print("\n 양성 vs 음성 강수 (양성 평균이 커야 정상임): ")
    for col in ["rain_1h", "rain_3h", "rain_24h"]:
        pos_mean = train_df.loc[train_df[TARGET_COL] == 1, col].mean()
        neg_mean = train_df.loc[train_df[TARGET_COL] == 0, col].mean()
        flag = "" if pos_mean >= neg_mean else " 라벨 반전 위험"
        print(f"  {col:>10s}: 양성={pos_mean:.2f}, 음성={neg_mean:.2f}{flag}")

    # ─────────────────────────────────────────
    # 3.  scale pos weight
    # ─────────────────────────────────────────
    n_pos = int(train_df[TARGET_COL].sum())
    n_neg = len(train_df) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)
    print(f"\nscale_pos_weight = {scale_pos_weight:.2f}")

    # ─────────────────────────────────────────
    # 4. Optuna sample
    # ─────────────────────────────────────────
    param_path = "best_params.json"
    if not os.path.exists(param_path):
        print("\nRunning Optuna...")
        sample_df = stratified_sample(train_df, n_total=500_000)
        print(f"  optuna sample size: {len(sample_df):,}")
        best_params = run_optuna(sample_df, scale_pos_weight=scale_pos_weight)
        del sample_df
        with open(param_path, "w") as f:
            json.dump(best_params, f)
    else:
        print("Loading saved parameters...")
        with open(param_path, "r") as f:
            best_params = json.load(f)

    # ─────────────────────────────────────────
    # 5. TRAIN
    # ─────────────────────────────────────────
    print("\n[Train] 최종 학습")
    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL].astype("int8")
    val_size = max(int(len(X_train) * 0.05), 5000)
    X_tr, X_val = X_train.iloc[:-val_size], X_train.iloc[-val_size:]
    y_tr, y_val = y_train.iloc[:-val_size], y_train.iloc[-val_size:]

    final_params = {
        "objective": "binary",
        "metric": "average_precision",
        "verbosity": -1,
        "n_estimators": 1000,
        "scale_pos_weight": scale_pos_weight,
        **best_params,
    }
    model = lgb.LGBMClassifier(**final_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                  lgb.log_evaluation(100)],
    )
    del train_df, X_train, y_train, X_tr, y_tr, X_val, y_val

    model.booster_.save_model(MODEL_PATH)
    print(f"saving model: {MODEL_PATH}")

    # ─────────────────────────────────────────
    # 6. TEST (streaming)
    # ─────────────────────────────────────────
    y_test, proba = evaluate_streaming(model, year_map, test_years)

    print("\nROC-AUC:", roc_auc_score(y_test, proba))
    print("PR-AUC :", average_precision_score(y_test, proba))

    # ─────────────────────────────────────────
    # 7. threshold (recall 중심) : thr 여러 기준으로 바꿔서 평가하도록
    # ─────────────────────────────────────────
    best_thresh = find_threshold_for_recall(y_test, proba, 0.6)
    print("Threshold:", best_thresh)

    pred = (proba >= best_thresh).astype(int)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, pred))

    print("\nPrecision:", precision_score(y_test, pred))
    print("Recall   :", recall_score(y_test, pred))
    print("F1       :", f1_score(y_test, pred))


if __name__ == "__main__":
    main()