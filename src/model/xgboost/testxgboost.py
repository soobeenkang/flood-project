import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb

# =========================
# 1. 데이터 로드
# =========================
df = pd.read_parquet("data/final/event_sampled_dataset_2pass.parquet")

# =========================
# 2. feature / label 분리
# =========================
exclude_cols = {
    "flood",
    "y",
    "time",
    "is_rainy_event",
    "rain_1h",
    "rain_3h",
    "rain_6h",
    "rain_12h",
    "rain_24h",
    "rain_intensity",
    "rain_max_3h",
    "water_level",
}

feature_cols = [c for c in df.columns if c not in exclude_cols]

X = df[feature_cols]
y = df["y"]

print("X shape:", X.shape)

# =========================
# 3. train/test split (event leakage 방지)
# =========================
# 시간 기준 split
df = df.sort_values("time")

split_idx = int(len(df) * 0.8)

train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

X_train = train_df[feature_cols]
y_train = train_df["y"]

X_test = test_df[feature_cols]
y_test = test_df["y"]

# =========================
# 4. XGBoost 모델
# =========================
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    tree_method="hist",   # CPU 빠른 버전
    random_state=42
)

# =========================
# 5. 학습
# =========================
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)

# =========================
# 6. 평가
# =========================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== ROC-AUC ===")
print(roc_auc_score(y_test, y_prob))