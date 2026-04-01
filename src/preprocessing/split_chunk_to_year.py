"""
    final chunk들을 병렬로 
    읽어서 연도별로 나누어 저장하는 코드

    최종: data/final_by_year/temp_{year}_{file}.parquet
"""
import pandas as pd
import os
from multiprocessing import Pool, cpu_count

input_dir = "data/tmp_final"
output_dir = "data/final_by_year"
os.makedirs(output_dir, exist_ok=True)


def process_file(file):

    df = pd.read_parquet(f"{input_dir}/{file}")
    df["time"] = pd.to_datetime(df["time"])
    df["year"] = df["time"].dt.year

    results = []

    for year, year_df in df.groupby("year"):
        temp_path = f"{output_dir}/temp_{year}_{file}.parquet"
        year_df.to_parquet(temp_path)
        results.append(temp_path)

    return results


def run_parallel():
    files = sorted(os.listdir(input_dir))

    with Pool(cpu_count() - 1) as p:
        p.map(process_file, files)
    
    print("청크 연도별 나누기 완료")


if __name__ == "__main__":
    run_parallel()