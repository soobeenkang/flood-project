"""
    침수 label 생성 및
    모델 학습을 위한 데이터 셋을 모두 merge 하는 코드
"""
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import os
import glob


"""
    aws 강수 제외 모든 데이터 준비
"""
# 하수구 수위, 지형, 세곡천 거리 데이터 준비
water = pd.read_parquet("data/final/gangnam_sewer_historical_grid.parquet")
elev = pd.read_parquet("data/final/gangnam_grid_elevation.parquet")
dist = pd.read_parquet("data/final/segokcheon_distance_class.parquet")
water["time"] = pd.to_datetime(water["time"])
water["grid_id"] = water["grid_id"].astype("int32")

# flood label 벡터화 생성
flood = pd.read_parquet("data/final/gangnam_final_flood_grid.parquet")
flood["SAT_DATE"] = pd.to_datetime(flood["SAT_DATE"])
flood["END_DATE"] = pd.to_datetime(flood["END_DATE"])
flood = flood[flood["IS_FLOODED"] == 1].copy()

# END DATE 없으면 당일 자정으로 셋
flood["END_DATE"] = flood["END_DATE"].fillna(
    flood["SAT_DATE"].dt.normalize() + pd.Timedelta(days=1)
)

# 행 시간 단위로 나누기
flood_rows = []
for _, row in flood[["grid_id", "SAT_DATE", "END_DATE"]].itertuples(index=False):
    hours = pd.date_range(row.SAT_DATE.floor("h"), row.END_DATE.floor("h"), freq="h")
    flood_rows.append(pd.DataFrame({"grid_id": row.grid_id, "time": hours}))

flood_map = pd.concat(flood_rows, ignore_index=True)
flood_map["IS_FLOODED"] = np.int8(1)
flood_map["grid_id"] = flood_map["grid_id"].astype("int32")

print(f"flood_map 크기: {len(flood_map)}")


"""
    aws 강수량 청크 단위로 읽고 처리
"""
os.makedirs("data/final/tmp_chunks", exist_ok=True)

parquet_file = pq.ParquetFile("data/final/grid_rainfall.parquet")

for i, batch in enumerate(parquet_file.iter_batches(batch_size=5_000_000)):

    print(f"청크 {i} 처리 중...")

    chunk = batch.to_pandas()
    chunk["grid_id"] = chunk["grid_id"].astype("int32")

    # merge
    chunk = chunk.merge(water, on=["time", "grid_id"], how="left")
    chunk = chunk.merge(elev, on="grid_id", how="left")
    chunk = chunk.merge(dist, on="grid_id", how="left")
    chunk = chunk.merge(flood_map, on=["grid_id", "time"], how="left")
    chunk["IS_FLOODED"] = chunk["IS_FLOODED"].fillna(0).astype(int8)

    chunk.to_parquet(
        f"data/final/tmp_chunks/chunk_{i}.parquet",
        engine="pyarrow",
        compression="snappy"
    )

print("청크 처리 끝")


"""
    청크 다 합치기
"""
files = sorted(glob.glob("data/final/tmp_chunks/chunk_*.parquet"))
writer = None

for f in files:
    table = pq.read_table(f)
    if writer is None:
        writer = pq.ParquetWriter(
            "data/final/gangnam_final_dataset.parquet",
            table.schema,
            compression="snappy"
        )
    writer.write_table(table)

if writer:
    writer.close()

print("최종 강남 데이터셋 생성 완료!!")


"""
    데이터 확인
"""
df = pd.read_parquet("data/final/gangnam_final_dataset.parquet")

print("=== 전체 shape ===")
print(df.shape)

print("\n=== water_level 결측치 비율 ===")
print(f"{df['water_level'].isnull().mean()*100:.2f}%")

print("\n=== 클래스 불균형 ===")
vc = df["IS_FLOODED"].value_counts()
print(vc)
print(f"침수 비율: {vc[1]/len(df)*100:.4f}%")

print("\n=== 침수 시 강수량 분포 ===")
print(df[df["IS_FLOODED"] == 1]["rain_1h"].describe())