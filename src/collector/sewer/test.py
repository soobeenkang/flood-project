import pandas as pd
his_sewer = pd.read_parquet("flood-project/data/output/seoul_sewer_historical_grid.parquet")
sewer = pd.read_parquet("flood-project/data/output/seoul_sewer_api.parquet")
flood = pd.read_parquet("flood-project/data/flood/seoul_final_flood_grid.parquet")


# 기본 데이터 체크 함수 
def check_df(df, name):
    print(f"\n===== {name} 기본 정보 =====")
    print(df.dtypes)
    print(df.shape)

    print(f"\n===== {name} 상위 데이터 =====")
    print(df.head())

    print(f"\n===== {name} 결측치 =====")
    print(df.isnull().sum())

    print(f"\n===== {name} 중복 =====")
    print(df.duplicated().sum())

    if "time" in df.columns:
        print(f"\n===== {name} 시간 범위 =====")
        print(df["time"].min(), df["time"].max())

    if "grid_id" in df.columns:
        print(f"\n===== {name} grid 개수 =====")
        print(df["grid_id"].nunique())


# 데이터 상태 확인
check_df(his_sewer, "historical")
check_df(sewer, "api")
check_df(flood, "flood")


# 침수 grid 분석 
print("\n================ 침수 분석 ================\n")

# 침수된 grid만 필터링
flooded = flood[flood["IS_FLOODED"] == 1]
print("침수 grid 수:", len(flooded))
print("----------------------------")

# 겹치는 grid 찾기
common = pd.merge(his_sewer, flooded, on="grid_id")
count = pd.merge(sewer, flooded, on="grid_id")

#  개수
print("겹치는 historical 침수 grid 개수:", common["grid_id"].nunique())
print("겹치는 api 침수 grid 개수:", count["grid_id"].nunique())


print("\n------------------------------------------------------")

# ===== grid 커버리지 비교 =====
his_ids = set(his_sewer["grid_id"].dropna().unique())
api_ids = set(sewer["grid_id"].dropna().unique())

overlap = his_ids & api_ids

print("historical grid 개수:", len(his_ids))
print("api grid 개수:", len(api_ids))
print("겹치는 grid 개수:", len(overlap))


#추가적으로 

# 침수 grid 중 historical에 없는 것
missing_in_his = set(flooded["grid_id"]) - his_ids
print("\n historical에 없는 침수 grid:", len(missing_in_his))

# 침수 grid 중 api에 없는 것
missing_in_api = set(flooded["grid_id"]) - api_ids
print(" api에 없는 침수 grid:", len(missing_in_api))
