import glob
import os
from pathlib import Path

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
    TEST_YEARS, N_TRIALS
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
# (옵션) Optuna용 샘플
# ───────────────────────────────────────────────
def load_small_sample(year_map, train_years, frac=0.1):
    df = pd.read_parquet(year_map[train_years[0]], columns=LOAD_COLS)
    return df.sample(frac=frac, random_state=42)


# ───────────────────────────────────────────────
# Optuna
# ───────────────────────────────────────────────
def run_optuna(sample_df):
    X = sample_df[FEATURE_COLS]
    y = sample_df[TARGET_COL]

    tscv = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "average_precision",
            "verbosity": -1,

            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "num_leaves": trial.suggest_int("num_leaves", 31, 128),
            "max_depth": trial.suggest_int("max_depth", 5, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),

            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 0.9),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 0.9),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 5),

            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),

            "n_estimators": 300,
        }

        scores = []

        for tr_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr, y_tr)

            proba = model.predict_proba(X_val)[:, 1]
            scores.append(average_precision_score(y_val, proba))

        return np.mean(scores)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42)
    )

    study.optimize(objective, n_trials=N_TRIALS)

    return study.best_params


# ───────────────────────────────────────────────
# threshold (recall 중심)
# ───────────────────────────────────────────────
def find_threshold_for_recall(y_true, proba, target_recall=0.6):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)

    idx = np.where(recall >= target_recall)[0]
    if len(idx) == 0:
        return 0.5

    best = idx[np.argmax(precision[idx])]
    return thresholds[min(best, len(thresholds) - 1)]


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
    # 1. Optuna sample
    # ─────────────────────────────────────────
    sample_df = load_small_sample(year_map, train_years)

    best_params = run_optuna(sample_df)
    del sample_df

    # ─────────────────────────────────────────
    # 2. TRAIN (NO sampling, NO weight)
    # ─────────────────────────────────────────
    model = None

    for y in train_years:
        print(f"[TRAIN] {y}")

        df = pd.read_parquet(year_map[y], columns=LOAD_COLS)

        X = df[FEATURE_COLS]
        y_ = df[TARGET_COL].astype("int8")

        if model is None:
            model = lgb.LGBMClassifier(
                **best_params,
                n_estimators=500
            )
            model.fit(X, y_)
        else:
            model.fit(X, y_, init_model=model)

        del df, X, y_

    # ─────────────────────────────────────────
    # 3. TEST (streaming)
    # ─────────────────────────────────────────
    y_test, proba = evaluate_streaming(model, year_map, test_years)

    print("\nROC-AUC:", roc_auc_score(y_test, proba))
    print("PR-AUC :", average_precision_score(y_test, proba))

    # ─────────────────────────────────────────
    # 4. threshold (recall 중심)
    # ─────────────────────────────────────────
    best_thresh = find_threshold_for_recall(y_test, proba, 0.9)
    print("Threshold:", best_thresh)

    pred = (proba >= best_thresh).astype(int)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, pred))

    print("\nPrecision:", precision_score(y_test, pred))
    print("Recall   :", recall_score(y_test, pred))
    print("F1       :", f1_score(y_test, pred))


if __name__ == "__main__":
    main()