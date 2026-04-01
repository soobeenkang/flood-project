"""
    aws 강수량 chunk 파일들과
    flood, sewer, elev, river 데이터를 머지해서
    final chunck 파일로 내보내는 코드

    최종: data/tmp_final/final_{i}.parquet
"""
import pandas as pd
import os

def build_final_dataset_from_chunks():

    #   flood 읽어오기
    flood = pd.read_parquet("data/final/seoul_final_flood_grid.parquet")
    flood["F_SAT_YMD"] = pd.to_datetime(flood["F_SAT_YMD"])
    flood["F_END_YMD"] = pd.to_datetime(flood["F_END_YMD"])
    flood = flood[
        flood["F_SAT_YMD"].notna() &
        flood["F_END_YMD"].notna()
    ]

    #   flood data 한시간 간격으로 펼치기
    flood_rows = []
    for row in flood.itertuples(index=False):
        times = pd.date_range(
            row.F_SAT_YMD.floor("h"),
            row.F_END_YMD.floor("h"),
            freq="h"
        )
        for t in times:
            flood_rows.append((row.grid_id, t, 1))

    flood_df = pd.DataFrame(flood_rows, columns=["grid_id", "time", "flood"])
    flood_df["flood"] = flood_df["flood"].astype("int8")

    print("flood_df:", flood_df.shape)

    #   water 읽어오기
    sewer = pd.read_parquet("data/final/seoul_sewer_historical_grid.parquet")
    sewer["time"] = pd.to_datetime(sewer["time"])
    #   elevation 읽어오기
    elev = pd.read_parquet("data/final/seoul_grid_with_elevation.parquet")
    #   river 읽어오기
    river = pd.read_parquet("data/final/seoul_grid_with_river_flag.parquet")
    river["is_river"] = river["is_river"].astype("int8")

    print("static 데이터 로딩 완료")

    #   output 만들기
    os.makedirs("data/tmp_final", exist_ok=True)

    files = sorted(os.listdir("data/tmp_rainfall_parquet"))
    
    for i, file in enumerate(files):
        if i % 10 == 0:
            print(f"\n{i} 처리 중: {file}")

        #   강수량 chunk 로딩
        df = pd.read_parquet(f"data/tmp_rainfall_parquet/{file}")
        df["time"] = pd.to_datetime(df["time"])

        #   flood merge
        df = df.merge(flood_df, on=["grid_id", "time"], how="left")
        df["flood"] = df["flood"].fillna(0).astype("int8")
        #   sewer merge
        df = df.merge(sewer, on=["grid_id", "time"], how="left")
        #   elev merge
        df = df.merge(elev, on=["grid_id"], how="left")
        #   river merge
        df = df.merge(river, on=["grid_id"], how="left")

        #   sorting values
        df = df.sort_values(["time", "grid_id"]).reset_index(drop=True)

        df.to_parquet(
            f"data/tmp_final/final_{i}.parquet",
            engine="pyarrow",
            compression="snappy"
        )
    
    print("전체 처리 완료")


if __name__ == "__main__":
    build_final_dataset_from_chunks()