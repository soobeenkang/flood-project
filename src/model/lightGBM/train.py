"""
학습/튜닝/평가 스크립트 (캐시된 피처를 입력으로).

전제: build_features.py가 먼저 실행되어 cache_features/feat_YYYY.parquet들이 존재해야 함.

언제 실행?
- 하이퍼파라미터 튜닝 반복
- threshold/SMOTE 옵션 비교
- 다른 분할 시도
- 모델 알고리즘 비교

사용:
    python train.py                          # 기본 설정으로 학습
    python train.py --n-trials 30            # Optuna trial 줄이기
    python train.py --skip-optuna            # 저장된 best_params.json 재사용
    python train.py --threshold 0.4          # 평가 임계값 변경
"""

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix, f1_score,
)
from imblearn.over_sampling import SMOTE
import warnings

from config import (
    CACHE_DIR, CACHE_PREFIX, FEATURE_COLS, TARGET_COL, LOAD_COLS,
    TEST_YEARS, N_TRIALS, CV_FOLDS, RANDOM_STATE,
    DEFAULT_THRESHOLD, MODEL_PATH, BEST_PARAMS_PATH,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ───────────────────────────────────────────────
# 캐시 디스커버리 / 로드
# ───────────────────────────────────────────────
def discover_cached_years() -> dict:
    """캐시 폴더에서 연도 → 경로 매핑 생성."""
    pattern = os.path.join(CACHE_DIR, f"{CACHE_PREFIX}*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"캐시 없음: {pattern}\n먼저 `python build_features.py` 실행 필요"
        )

    year_to_path = {}
    for f in files:
        stem = Path(f).stem  # feat_2010
        year = int(stem.replace(CACHE_PREFIX, ""))
        year_to_path[year] = f
    return year_to_path


def load_years(year_to_path: dict, years: list, columns: list = None) -> pd.DataFrame:
    """지정 연도 캐시들을 결합."""
    dfs = []
    for y in sorted(years):
        if y not in year_to_path:
            print(f"  경고: {y} 캐시 없음, 스킵")
            continue
        dfs.append(pd.read_parquet(year_to_path[y], columns=columns))
    return pd.concat(dfs, ignore_index=True)


# ───────────────────────────────────────────────
# SMOTE
# ───────────────────────────────────────────────
def apply_smote(X_train, y_train, random_state=RANDOM_STATE, verbose=True):
    if verbose:
        print(f"SMOTE 전 - 침수: {y_train.sum():,}, 정상: {(y_train==0).sum():,}")
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    if verbose:
        print(f"SMOTE 후 - 침수: {y_res.sum():,}, 정상: {(y_res==0).sum():,}")
    return X_res, y_res


# ───────────────────────────────────────────────
# Optuna 튜닝
# ───────────────────────────────────────────────
def run_optuna(X_train, y_train, n_trials=N_TRIALS, cv_folds=CV_FOLDS):
    tscv = TimeSeriesSplit(n_splits=cv_folds)

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "average_precision",
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
        for tr_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            if y_tr.sum() < 5 or y_val.sum() == 0:
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
            scores.append(average_precision_score(y_val, proba))

        return np.mean(scores) if scores else 0.0

    study = optuna.create_study(
        direction="maximize", sampler=TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"\nBest PR-AUC (CV): {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study.best_params


# ───────────────────────────────────────────────
# 학습 / 평가
# ───────────────────────────────────────────────
def train_final_model(best_params, X_train, y_train, X_test, y_test):
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


def evaluate(model, X_test, y_test, threshold=DEFAULT_THRESHOLD):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)

    print("\n" + "=" * 50)
    print(f"평가 임계값: {threshold}")
    print("=" * 50)
    print(classification_report(y_test, pred, target_names=["정상(0)", "침수(1)"], digits=4))
    print(f"ROC-AUC          : {roc_auc_score(y_test, proba):.4f}")
    print(f"PR-AUC (침수=1)  : {average_precision_score(y_test, proba):.4f}")
    print(f"PR-AUC (정상=0)  : {average_precision_score(1 - y_test, 1 - proba):.4f}")
    print(f"F1 (정상=0)      : {f1_score(y_test, pred, pos_label=0):.4f}")
    print(f"F1 (침수=1)      : {f1_score(y_test, pred, pos_label=1):.4f}")
    print(f"F1 (macro)       : {f1_score(y_test, pred, average='macro'):.4f}")

    cm = confusion_matrix(y_test, pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n혼동행렬:")
    print(f"            예측 정상    예측 침수")
    print(f"실제 정상   {tn:>10d}    {fp:>10d}")
    print(f"실제 침수   {fn:>10d}    {tp:>10d}")
    print(f"\n특이도(정상 재현율): {tn/(tn+fp):.4f}")
    print(f"민감도(침수 재현율): {tp/(tp+fn):.4f}")

    precision, recall, thresholds = precision_recall_curve(y_test, proba)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_thresh = thresholds[np.argmax(f1_scores[:-1])]
    print(f"\n최적 임계값 (침수 F1 최대): {best_thresh:.3f}")
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
# 메인
# ───────────────────────────────────────────────
def main(test_years=TEST_YEARS, n_trials=N_TRIALS,
         threshold=DEFAULT_THRESHOLD, skip_optuna=False):
    print("=" * 50)
    print("  도시 침수 예측 - 학습 파이프라인")
    print("=" * 50)

    # 1. 캐시 디스커버리
    year_to_path = discover_cached_years()
    all_years = sorted(year_to_path.keys())
    print(f"사용 가능 연도: {all_years}")

    # 2. 연도 기반 분할
    train_years = all_years[:-test_years]
    test_years_list = all_years[-test_years:]
    print(f"Train 연도: {train_years}")
    print(f"Test  연도: {test_years_list}")

    # 3. 캐시 로드 (필요 컬럼만)
    train_df = load_years(year_to_path, train_years, columns=LOAD_COLS)
    test_df = load_years(year_to_path, test_years_list, columns=LOAD_COLS)

    print(f"\nTrain: {train_df.shape}, 침수 비율: {train_df[TARGET_COL].mean():.4f}")
    print(f"Test : {test_df.shape}, 침수 비율: {test_df[TARGET_COL].mean():.4f}")

    available = [c for c in FEATURE_COLS if c in train_df.columns]
    X_train = train_df[available].reset_index(drop=True)
    y_train = train_df[TARGET_COL].astype("int8").reset_index(drop=True)
    X_test = test_df[available].reset_index(drop=True)
    y_test = test_df[TARGET_COL].astype("int8").reset_index(drop=True)
    del train_df, test_df

    # 4. Optuna 튜닝 (선택적 스킵)
    if skip_optuna and os.path.exists(BEST_PARAMS_PATH):
        with open(BEST_PARAMS_PATH, "r") as f:
            best_params = json.load(f)
        print(f"\n[skip-optuna] 저장된 파라미터 사용: {BEST_PARAMS_PATH}")
        print(f"  {best_params}")
    else:
        print(f"\nOptuna 튜닝 시작 ({n_trials} trials)...")
        best_params = run_optuna(X_train, y_train, n_trials=n_trials)
        with open(BEST_PARAMS_PATH, "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"파라미터 저장: {BEST_PARAMS_PATH}")

    # 5. 최종 학습
    print("\n최종 모델 학습용 SMOTE...")
    X_tr_s, y_tr_s = apply_smote(X_train, y_train)
    model = train_final_model(best_params, X_tr_s, y_tr_s, X_test, y_test)

    # 6. 평가
    proba, best_thresh = evaluate(model, X_test, y_test, threshold=threshold)
    feature_importance_report(model, available)

    '''
    # 7. 모델 저장
    model.booster_.save_model(MODEL_PATH)
    print(f"\n모델 저장: {MODEL_PATH}")
    '''

    return model, best_thresh, proba


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-years", type=int, default=TEST_YEARS)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--skip-optuna", action="store_true",
                        help=f"저장된 {BEST_PARAMS_PATH} 사용, Optuna 스킵")
    args = parser.parse_args()

    main(
        test_years=args.test_years,
        n_trials=args.n_trials,
        threshold=args.threshold,
        skip_optuna=args.skip_optuna,
    )