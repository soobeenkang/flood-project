import pandas as pd


# 시간 당 강수량을 바탕으로 누적 강수량 학습 데이터 셋 만들기
def create_rain_features():

    df = pd.read_csv("data/prev_rainfall/aws_hourly_raw.csv")

    df["time"] = pd.to_datetime(df["time"])

    df = df.sort_values(["station","time"])

    df["rain_3h"] = df.groupby("station")["rain_1h"].rolling(3).sum().reset_index(0,drop=True)
    df["rain_6h"] = df.groupby("station")["rain_1h"].rolling(6).sum().reset_index(0,drop=True)
    df["rain_12h"] = df.groupby("station")["rain_1h"].rolling(12).sum().reset_index(0,drop=True)
    df["rain_24h"] = df.groupby("station")["rain_1h"].rolling(24).sum().reset_index(0,drop=True)
    df["rain_intensity"] = df.groupby("station")["rain_1h"].rolling(3).mean().reset_index(0,drop=True)
    df["rain_max_3h"] = df.groupby("station")["rain_1h"].rolling(3).max().reset_index(0,drop=True)

    df.to_csv("data/prev_rainfall/aws_rainfall_features.csv", index=False)

    print("rain feature 생성 완료")


if __name__ == "__main__":
    create_rain_features()