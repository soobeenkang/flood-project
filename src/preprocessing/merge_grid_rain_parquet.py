"""
    만들어진 시간 당 parquet 파일을 합쳐서
    하나의 parquet으로 만드는 코드
"""
import pandas as pd
import glob
import pyarrow as pa
import pyarrow.parquet as pq


def merge_parquet():

    files = sorted(glob.glob("data/tmp_parquet/*.parquet"))
    print(f"파일 개수: {len(files)}")

    writer = None

    for f in files:
        df = pd.read_parquet(f)

        table = pa.Table.from_pandas(df)

        if writer is None:
            writer = pq.ParquetWriter(
                "data/final/grid_rainfall.parquet",
                table.schema,
                compression="snappy"
            )

        writer.write_table(table)

        if i % 1000 == 0:
            print(f"{i}/{len(files)} 완료")

    if writer:
        writer.close()

    print("최종 강남 aws rainfall with grid parquet 생성 완료")


if __name__ == "__main__":
    merge_parquet()