"""
    static data인 elevation, is_river를
    침수 이력이 있는 grid_id 기준으로 머지해
    하나의 파일로 저장하는 코드
    최종: grid_id, elevation, is_river
"""
import pandas as pd

def merge_static_data():
    # flood 읽어와서 침수 이력 있는 그리드만 남겨두기
    flood = pd.read_parquet("data/final/seoul_final_flood_grid.parquet")
    flood["F_SAT_YMD"] = pd.to_datetime(flood["F_SAT_YMD"])
    flood["F_END_YMD"] = pd.to_datetime(flood["F_END_YMD"])
    flood = flood[
        flood["F_SAT_YMD"].notna() &
        flood["F_END_YMD"].notna()
    ]

    flood_grids = flood["grid_id"].unique()

    elev = pd.read_parquet("data/final/seoul_grid_with_elevation.parquet")
    river = pd.read_parquet("data/final/seoul_grid_with_river_flag.parquet")
    river["is_river"] = river["is_river"].astype("int8")

    elev = elev[elev["grid_id"].isin(flood_grids)]

    elev = elev.merge(river, on=["grid_id"], how="left")

    elev.sort_values(["grid_id"]).reset_index(drop=True)

    elev.to_parquet(
        f"data/final/static_data.parquet",
        engine="pyarrow",
        compression="snappy"
    )

    print("elev, river 머지 완료")

if __name__ == "__main__":
    merge_static_data()