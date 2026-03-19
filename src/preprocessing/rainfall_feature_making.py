import pandas as pd


# 시간 당 강수량을 바탕으로 누적 강수량 학습 데이터 셋 만들기
def create_rain_features():

    df = pd.read_csv("data/prev_rainfall/aws_hourly_raw.csv")

    df["time"] = pd.to_datetime(df["time"])

    df = df.sort_values(["station","time"]).reset_index(drop=True)

    df = df.set_index("time")

    df["rain_3h"] = df.groupby("station")["rain_1h"].transform(
        lambda x: x.rolling("3h", min_periods=1).sum()
    )
    df["rain_6h"] = df.groupby("station")["rain_1h"].transform(
        lambda x: x.rolling("6h", min_periods=1).sum()
    )
    df["rain_12h"] = df.groupby("station")["rain_1h"].transform(
        lambda x: x.rolling("12h", min_periods=1).sum()
    )
    df["rain_24h"] = df.groupby("station")["rain_1h"].transform(
        lambda x: x.rolling("12h", min_periods=1).sum()
    )
    df["rain_intensity"] = df.groupby("station")["rain_1h"].transform(
        lambda x: x.rolling("3h", min_periods=1).mean()
    )
    df["rain_max_3h"] = df.groupby("station")["rain_1h"].transform(
        lambda x: x.rolling("3h", min_periods=1).max()
    )

    df = df.reset_index()

    df.to_csv("data/prev_rainfall/aws_rainfall_features.csv", index=False)

    print("rain feature 생성 완료")


if __name__ == "__main__":
    create_rain_features()