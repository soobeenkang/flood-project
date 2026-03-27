"""
    aws 강수량 관측 데이터 csv 파일 읽어서
    2010-2024년의 5-10월 사이의 시간, 관측소, 강수량을 필터해
    일시에 맞는 관측소의 위경도와 함께 csv로 저장하는 코드
"""

import pandas as pd
import glob
import zipfile
import os

# zip 해제하는 함수
def unzip_all(base_path):
    while True:
        found_zip = False

        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.endswith(".zip"):
                    found_zip = True
                    zip_path = os.path.join(root, file)
                    extract_path = os.path.splitext(zip_path)[0]

                    try:
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(extract_path)
                        os.remove(zip_path)

                    except Exception as e:
                        print(f"압축 해제 실패: {zip_path} -> {e}")
            
        if not found_zip:
            break


# csv 파일 읽어오는 함수
def load_aws_files():

    files = glob.glob("data/raw_aws/unzipped/**/*.csv", recursive=True)
    print("csv 파일 개수:", len(files))

    dfs = []

    # 필요한 컬럼 (지점, 지점명, 일시, 강수량) 만 읽어오기
    for f in files:
        df = pd.read_csv(f, encoding="cp949", usecols=["지점", "일시", "강수량(mm)"])
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    return df


#   2010-2024 5-10월의 필요한 컬럼만 추출 & 사이즈 줄이는 함수
#   해당 일시의 station lat, lon 병합
def preprocess():

    unzip_all("data/raw_aws/unzipped")
    df = load_aws_files()

    df = df.rename(columns={
        "지점":"station",
        "일시":"time",
        "강수량(mm)":"rain_1h"
    })

    df["time"] = pd.to_datetime(df["time"])

    # --- 2010-2024년의 5-10월로 데이터 제한 ---
    df = df[df["time"].dt.month.between(5,10)]
    df = df[df["time"].dt.year.between(2010, 2024)]

    df = df.sort_values(["station","time"])

    df = df[["time","station","rain_1h"]]

    # 사이즈 줄이기
    df["station"] = df["station"].astype("int16")
    df["rain_1h"] = df["rain_1h"].astype("float32")


    # --- 일시에 맞는 관측소 좌표 병합 ---
    meta = pd.read_csv("data/raw_aws/station_coords.csv")

    meta["start_date"] = pd.to_datetime(meta["start_date"])
    meta["expired_date"] = pd.to_datetime(meta["expired_date"])
    # expired_date 없을 경우 미래로 설정
    meta["expired_date"] = meta["expired_date"].fillna(pd.Timestamp("2100-01-01"))
    
    meta["station"] = meta["station"].astype("int16")
    # time 기준으로 가장 가까운 start_date 붙임 & expired_date 만 체크
    df = df.sort_values("time")
    meta = meta.sort_values("start_date")

    df = pd.merge_asof(
        df,
        meta,
        by="station",
        left_on="time",
        right_on="start_date",
        direction="backward"
    )
    df = df[df["time"] < df["expired_date"]]

    df = df[["time", "station", "rain_1h", "lat", "lon"]]

    df.to_csv("data/rainfall_history/aws_hourly_raw.csv",index=False)

    print("AWS raw rainfall  생성 완료")


if __name__ == "__main__":
    preprocess()