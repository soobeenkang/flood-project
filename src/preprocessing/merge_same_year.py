"""
    연도별 파일들을 하나의 연도 파일로 합치는 코드
    
    최종: data/final_by_year_merged/final_{year}.parquet
"""
import pandas as pd
import os
from glob import glob

def merge_year_files():
    input_dir = "data/final_by_year"
    output_dir = "data/final_by_year_merged"
    os.makedirs(output_dir, exist_ok=True)

    years = set()

    for file in os.listdir(input_dir):
        if file.startswith("temp_"):
            year = file.split("_")[1]
            years.add(year)

    for year in sorted(years):
        print(f"{year} 합치는 중")

        files = glob(f"{input_dir}/temp_{year}_*.parquet")

        dfs = []
        for f in files:
            dfs.append(pd.read_parquet(f))

        final_df = pd.concat(dfs, ignore_index=True)

        final_df.to_parquet(
            f"{output_dir}/final_{year}.parquet",
            engine="pyarrow",
            compression="snappy"
        )

        # 병합 끝난 연도별 tmp 파일들은 삭제
        for f in files:
            os.remove(f)
    
    print("연도별 단일 파일 생성 완료")

if __name__ == "__main__":
    merge_year_files()