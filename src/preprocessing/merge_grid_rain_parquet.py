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
                "data/prev_rainfall/grid_rainfall.parquet",
                table.schema,
                compression="snappy"
            )

        writer.write_table(table)

    if writer:
        writer.close()

    print("최종 aws rainfall with grid parquet 생성 완료")


if __name__ == "__main__":
    merge_parquet()