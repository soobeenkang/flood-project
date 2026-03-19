import pandas as pd
import glob
import zipfile
import os

STATIONS = [400,401,403,413,414]

def load_aws_files():

    files = glob.glob("data/raw_aws/unzipped/*.csv", recursive=True)

    print("csv 파일 개수:", len(files))

    dfs = []

    for f in files:

        df = pd.read_csv(f, encoding="cp949")

        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    return df


def preprocess():

    df = load_aws_files()

    df = df.rename(columns={
        "지점":"station",
        "지점명":"station_name",
        "일시":"time",
        "강수량(mm)":"rain_1h"
    })

    df["time"] = pd.to_datetime(df["time"])

    # 5-10월로 데이터 제한
    df["month"] = df["time"].dt.month
    df = df[df["month"].between(5,10)]

    df = df[df["station"].isin(STATIONS)]

    df = df.sort_values(["station","time"])

    df = df[["time","station","rain_1h"]]

    df.to_csv("data/prev_rainfall/aws_hourly_raw.csv",index=False)

    print("AWS raw rainfall  생성 완료")


if __name__ == "__main__":
    preprocess()