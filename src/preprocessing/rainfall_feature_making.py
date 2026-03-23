"""
    1시간 강수량을 사용해 누적 강수량(1,3,6,12,24h)과
    강수정도, 3시간 최대 강수량을 구하는 코드
"""

import pandas as pd

def create_rain_features():

    df = pd.read_csv("data/rainfall_history/aws_hourly_raw.csv")

    df["time"] = pd.to_datetime(df["time"])

    df = df.sort_values(["station","time"]).reset_index(drop=True)

    results = []

    # station으로 나눠서 처리 for memory
    for station_id, group in df.groupby("station"):
        group = group.set_index("time")

        group["rain_3h"] = group["rain_1h"].transform(
        lambda x: x.rolling("3h", min_periods=1).sum()
    )
        group["rain_6h"] = group["rain_1h"].transform(
        lambda x: x.rolling("6h", min_periods=1).sum()
    )
        group["rain_12h"] = group["rain_1h"].transform(
        lambda x: x.rolling("12h", min_periods=1).sum()
    )
        group["rain_24h"] = group["rain_1h"].transform(
        lambda x: x.rolling("24h", min_periods=1).sum()
    )
        group["rain_intensity"] = group["rain_1h"].transform(
        lambda x: x.rolling("3h", min_periods=1).mean()
    )
        group["rain_max_3h"] = group["rain_1h"].transform(
        lambda x: x.rolling("3h", min_periods=1).max()
    )
    
        group = group.reset_index()

        # float32로 사이즈 줄이기
        for col in ["rain_3h", "rain_6h", "rain_12h", "rain_24h", "rain_intensity", "rain_max_3h"]:
            group[col] = group[col].astype("float32")

        results.append(group)

    df_out = pd.concat(results, ignore_index=True)

    df_out.to_csv("data/rainfall_history/aws_rainfall_features.csv", index=False)

    print("rain feature 생성 완료")


if __name__ == "__main__":
    create_rain_features()