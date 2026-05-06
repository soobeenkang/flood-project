import pandas as pd
import numpy as np
import lightgbm as lgb
from config import CACHE_DIR, CACHE_PREFIX, FEATURE_COLS, TARGET_COL, LOAD_COLS
from train import discover_cached_years  # 함수명이 train.py에 있다면

# 모델 로드
model = lgb.Booster(model_file="model.txt")

# test 연도 한 해 로드
year_map = discover_cached_years()
test_year = sorted(year_map.keys())[-1]  # 가장 최근 연도
print(f"진단 대상: {test_year}")

df_test = pd.read_parquet(year_map[test_year], columns=LOAD_COLS)

# 1. 강수 분포
print("\n--- test 강수 분포 ---")
for col in ["rain_1h", "rain_3h", "rain_24h"]:
    pos_mean = df_test.loc[df_test[TARGET_COL]==1, col].mean()
    neg_mean = df_test.loc[df_test[TARGET_COL]==0, col].mean()
    print(f"  {col}: 양성={pos_mean:.3f}, 음성={neg_mean:.3f}")

# 2. proba 분포
proba = model.predict(df_test[FEATURE_COLS])  # Booster는 predict가 바로 확률
y = df_test[TARGET_COL].values

print(f"\n--- proba 분포 ---")
print(f"  양성 평균: {proba[y==1].mean():.5f}, 중앙: {np.median(proba[y==1]):.5f}")
print(f"  음성 평균: {proba[y==0].mean():.5f}, 중앙: {np.median(proba[y==0]):.5f}")

# 3. feature importance도 같이 보기
print(f"\n--- 상위 feature ---")
imp = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": model.feature_importance(importance_type="gain")
}).sort_values("importance", ascending=False).head(15)
print(imp.to_string(index=False))