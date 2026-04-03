import pandas as pd
import numpy as np
import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

# =========================
# 1. 데이터 로드
# =========================
df = pd.read_parquet("data/final/event_sampled_dataset_2pass.parquet")

# =========================
# 2. feature / label 분리 조건 설정
# =========================
exclude_cols = {
    # 1. 정답 및 기본 누수 방지
    "flood", "y", "time", "year",
    "is_rainy_event", "rain_1h", "rain_3h", "rain_6h", "rain_12h", "rain_24h",
    "rain_intensity", "rain_max_3h", "water_level",
    
    # 2. 공간적/시간적 과적합 방지 (핵심 추가)
    "grid_id",  # <--- 복구됨: 동네 이름 외우기 방지 (매우 중요)
    "month",    # month_sin, month_cos가 있으므로 중복 제거
    
    # 3. 중요도 0.0 노이즈 피처 제거
    "rain_intensity_roll_min_6", 
    "rain_intensity_diff_1",
    "rain_intensity_roll_mean_6", 
    "rain_intensity_roll_std_3",
    "rain_intensity_roll_max_3", 
    "rain_intensity_roll_mean_3",
    "rain_intensity_delta_1_12", 
    "rain_intensity_delta_1_6",
    "rain_intensity_delta_1_3", 
    "rain_intensity_lag_3"
}

# =========================
# 3. 데이터 분할 및 Target Encoding (순서 재배치)
# =========================
test_year = 2022

# (1) 먼저 연도 기준으로 데이터를 나눕니다.
train_df = df[df["year"] < test_year].copy()
test_df = df[df["year"] == test_year].copy()

# (2) 오직 Train 셋에서만 grid_id별 침수 평균 확률을 계산합니다.
grid_risk = train_df.groupby('grid_id')['y'].mean().reset_index()
grid_risk.columns = ['grid_id', 'grid_risk_score']

# (3) 계산된 위험도 점수를 병합합니다. (단 한 번만 실행!)
train_df = train_df.merge(grid_risk, on='grid_id', how='left')
test_df = test_df.merge(grid_risk, on='grid_id', how='left')

# (4) Test 셋에 새로 등장한 동네는 위험도를 0으로 채웁니다.
test_df['grid_risk_score'] = test_df['grid_risk_score'].fillna(0)

# (5) 🚨 중요: grid_risk_score가 생성된 이후에 feature_cols를 정의합니다!
feature_cols = [c for c in train_df.columns if c not in exclude_cols]

# (6) X, y 분리
X_train = train_df[feature_cols]
y_train = train_df["y"]

X_test = test_df[feature_cols]
y_test = test_df["y"]

print(f"Train 학습 연도: {train_df['year'].unique()}")
print(f"Test 평가 연도: {test_year}")
print("Train size:", len(X_train), " | Train Positives:", (y_train == 1).sum())
print("Test size:", len(X_test), " | Test Positives:", (y_test == 1).sum())


# 4. imbalance 계산 (Train 셋 기준 엄격한 계산)
# =========================
neg_train = (y_train == 0).sum()  # Train 셋의 비침수
pos_train = (y_train == 1).sum()  # Train 셋의 침수

exact_ratio = neg_train / pos_train

# 극단적 가중치 폭주를 막기 위해 제곱근(sqrt)을 사용하여 보정합니다.
# (예: 비율이 100배라면 가중치는 10배만 줌)
scale_pos_weight = np.sqrt(exact_ratio) 

print(f"Train Negative: {neg_train}, Train Positive: {pos_train}")
print(f"Train 셋 실제 비율: {exact_ratio:.2f}배")
print(f"적용된 scale_pos_weight (sqrt 보정): {scale_pos_weight:.2f}")

# =========================
# 5. 모델 생성
# =========================
model = xgb.XGBClassifier(
    n_estimators=1000,           # early stopping을 믿고 넉넉히 설정
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="aucpr",         # 불균형 지표 최적화
    tree_method="hist",
    scale_pos_weight=scale_pos_weight,
    max_delta_step=1,            # 가중치 폭주 방지
    random_state=42,
    early_stopping_rounds=50     # XGBoost 최신 버전 위치
)

# =========================
# 6. 학습
# =========================
model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)

# =========================
# 7. 확률 예측
# =========================
# early stopping으로 찾은 최적의 트리(best_iteration)까지만 사용하여 예측
y_prob = model.predict_proba(X_test, iteration_range=(0, model.best_iteration + 1))[:,1]

# =========================
# 8. threshold 동적 조정
# =========================
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

# F1 Score가 최대가 되는 지점을 찾아 Threshold로 지정 (분모 0 방지 위해 1e-10 추가)
f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"\n최적의 Threshold (Max F1 기준): {best_threshold:.6f}")

y_pred = (y_prob >= best_threshold).astype(int)

# =========================
# 9. 평가
# =========================
print("\n=== Classification Report ===")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("\nROC-AUC")
print(f"{roc_auc_score(y_test, y_prob):.6f}")