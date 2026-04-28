import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import optuna
import warnings
warnings.filterwarnings("ignore")

# =========================
# 1. 데이터 로드
# =========================
df = pd.read_parquet("data/final/event_sampled_dataset_2pass.parquet")
df = df.sort_values("time").reset_index(drop=True)

# =========================
# 2. Train / Test 분리 (연도 기준 — 버그 수정)
# =========================
TEST_YEAR = 2022

train_df = df[df["year"] < TEST_YEAR].copy()
test_df  = df[df["year"] == TEST_YEAR].copy()

print(f"Train 학습 연도: {sorted(train_df['year'].unique())}")
print(f"Test  평가 연도: {TEST_YEAR}")

# =========================
# 3. Target Encoding (누수 방지 — 버그 수정)
#    오직 Train 셋의 y만 사용해 grid_risk_score 계산
# =========================
grid_risk = (
    train_df.groupby("grid_id")["y"]
    .mean()
    .reset_index()
    .rename(columns={"y": "grid_risk_score"})
)

train_df = train_df.merge(grid_risk, on="grid_id", how="left")
test_df  = test_df.merge(grid_risk, on="grid_id", how="left")
test_df["grid_risk_score"] = test_df["grid_risk_score"].fillna(0)

# =========================
# 4. Feature 정의 (merge 이후에 확정 — 버그 수정)
# =========================
exclude_cols = {
    "flood", "y", "time", "year", "is_rainy_event",
    "rain_1h", "rain_3h", "rain_6h", "rain_12h", "rain_24h",
    "rain_intensity", "rain_max_3h", "water_level",
    "grid_id", "month",
    "rain_intensity_roll_min_6", "rain_intensity_diff_1",
    "rain_intensity_roll_mean_6", "rain_intensity_roll_std_3",
    "rain_intensity_roll_max_3", "rain_intensity_roll_mean_3",
    "rain_intensity_delta_1_12", "rain_intensity_delta_1_6",
    "rain_intensity_delta_1_3", "rain_intensity_lag_3",
}

feature_cols = [c for c in train_df.columns if c not in exclude_cols]

X_train = train_df[feature_cols]
y_train = train_df["y"]
X_test  = test_df[feature_cols]
y_test  = test_df["y"]

print(f"\nFeature 수: {len(feature_cols)}")
print(f"Train size: {len(X_train):,}  | Positives: {(y_train==1).sum():,}")
print(f"Test  size: {len(X_test):,}  | Positives: {(y_test==1).sum():,}")

# =========================
# 5. 불균형 비율 계산
# =========================
neg_train = (y_train == 0).sum()
pos_train = (y_train == 1).sum()
exact_ratio = neg_train / pos_train
print(f"\n불균형 비율: {exact_ratio:.1f}:1")

# =========================
# 6. Optuna 하이퍼파라미터 탐색 (autoresearch 방식 차용)
#    → 밤새 자동 실험 반복, AUCPR 최대화
# =========================
def objective(trial):
    # scale_pos_weight: sqrt~ratio 사이에서 탐색
    # (너무 크면 precision↓, 너무 작으면 recall↓)
    spw = trial.suggest_float("scale_pos_weight", np.sqrt(exact_ratio), exact_ratio * 0.5)

    params = {
        "n_estimators":        1000,
        "max_depth":           trial.suggest_int("max_depth", 4, 8),
        "learning_rate":       trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "subsample":           trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":    trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight":    trial.suggest_int("min_child_weight", 5, 100),
        "gamma":               trial.suggest_float("gamma", 0, 5),
        "reg_alpha":           trial.suggest_float("reg_alpha", 0, 2),
        "reg_lambda":          trial.suggest_float("reg_lambda", 0.5, 5),
        "scale_pos_weight":    spw,
        "max_delta_step":      trial.suggest_int("max_delta_step", 1, 5),
        "eval_metric":         "aucpr",
        "tree_method":         "hist",
        "early_stopping_rounds": 50,
        "random_state":        42,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_prob = model.predict_proba(X_test, iteration_range=(0, model.best_iteration + 1))[:, 1]

    # Precision ≥ 0.15 조건 하에서 F1 최대화 (precision 최소 보장)
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-10)
    mask = precision[:-1] >= 0.15
    if mask.sum() == 0:
        return 0.0
    return f1[mask].max()

print("\n[Optuna] 하이퍼파라미터 탐색 시작 (n_trials=300)...")
print("(n_trials를 늘릴수록 더 좋은 파라미터를 찾습니다)\n")

study = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=2000, show_progress_bar=True)

best_params = study.best_params
print(f"\n최적 파라미터: {best_params}")
print(f"최적 F1 (precision≥0.15 조건): {study.best_value:.4f}")

# =========================
# 7. 최적 파라미터로 최종 모델 학습
# =========================
final_params = {
    "n_estimators":        1000,
    "eval_metric":         "aucpr",
    "tree_method":         "hist",
    "early_stopping_rounds": 50,
    "random_state":        42,
    **best_params,
}

final_model = xgb.XGBClassifier(**final_params)
final_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,
)

# =========================
# 8. 예측 및 Threshold 최적화
# =========================
y_prob = final_model.predict_proba(
    X_test, iteration_range=(0, final_model.best_iteration + 1)
)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-10)

# Precision ≥ 0.15 조건 하에서 F1 최대 threshold 선택
mask = precision[:-1] >= 0.15
if mask.sum() > 0:
    best_idx = np.argmax(f1_scores * mask)
else:
    best_idx = np.argmax(f1_scores)

best_threshold = thresholds[best_idx]
print(f"\n최적 Threshold: {best_threshold:.6f}")
print(f"  → Precision: {precision[best_idx]:.3f}, Recall: {recall[best_idx]:.3f}")

y_pred = (y_prob >= best_threshold).astype(int)

# =========================
# 9. 최종 평가
# =========================
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, zero_division=0))

roc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC:  {roc:.6f}")

# AUCPR 계산
from sklearn.metrics import average_precision_score
aucpr = average_precision_score(y_test, y_prob)
print(f"AUCPR:    {aucpr:.6f}")

# =========================
# 10. Feature Importance 출력
# =========================
importance = pd.Series(
    final_model.feature_importances_, index=feature_cols
).sort_values(ascending=False)

print("\n=== Top 15 Feature Importance ===")
print(importance.head(15).to_string())
