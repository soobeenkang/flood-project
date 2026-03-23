"""
    aws 강수량 관측 데이터 csv 파일 읽어서
    5-10월 사이의 시간, 관측소, 강수량을 csv로 저장하는 코드
"""

import pandas as pd
import glob

# 강남구 근처 5개 aws 관측소 코드
STATIONS = [400,401,403,413,414]

# csv 파일 읽어오는 함수
def load_aws_files():

    files = glob.glob("data/raw_aws/unzipped/*.csv", recursive=True)
    print("csv 파일 개수:", len(files))

    dfs = []

    # 필요한 컬럼 (지점, 지점명, 일시, 강수량) 만 읽어오기
    for f in files:
        df = pd.read_csv(f, encoding="cp949", usecols=["지점", "지점명", "일시", "강수량(mm)"])
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    return df


#   5-10월의 필요한 컬럼만 추출 & 사이즈 줄이는 함수
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

    # 사이즈 줄이기
    df["station"] = df["station"].astype("int16")
    df["rain_1h"] = df["rain_1h"].astype("float32")

    df.to_csv("data/rainfall_history/aws_hourly_raw.csv",index=False)

    print("AWS raw rainfall  생성 완료")


if __name__ == "__main__":
    preprocess()