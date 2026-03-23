"""
    그리드 셀과 시간 당 강수량을
    idw 함수를 사용해 매핑하는 코드
"""
import geopandas as gpd
import pandas as pd
import numpy as np
from numba import njit
from joblib import Parallel, delayed
import os

# 관측소 좌표
station_coords = {
    400:(127.047,37.517),
    401:(127.015,37.483),
    403:(127.113,37.514),
    413:(127.041,37.563),
    414:(126.951,37.512)
}

# 강수 feature
features = [
    "rain_1h",
    "rain_3h",
    "rain_6h",
    "rain_12h",
    "rain_24h",
    "rain_intensity",
    "rain_max_3h"
]

#   5개의 관측소와의 거리를 이용해 가중치로 해당 그리드 셀 강수 계산
@njit
def idw(grid_points, station_points, station_values):

    G = grid_points.shape[0]
    S = station_points.shape[0]

    result = np.zeros(G)

    for i in range(G):

        num = 0.0
        den = 0.0

        for j in range(S):

            dx = grid_points[i, 0] - station_points[j, 0]
            dy = grid_points[i, 1] - station_points[j, 1]

            dist = (dx*dx + dy*dy) ** 0.5

            if dist == 0:
                dist = 1e-6

            w = 1.0 / (dist*dist)

            num += w * station_values[j]
            den += w

        result[i] = num / den

    return result


# parquet 파일 만드는 함수
def generate_grid_rain_parquet():

    grid = gpd.read_file("data/grid/gangnam_grid.geojson")

    if "grid_id" not in grid.columns:
        grid = grid.reset_index().rename(columns={"index": "grid_id"})

    rain = pd.read_csv("data/prev_rainfall/aws_rainfall_features.csv")
    rain["time"] = pd.to_datetime(rain["time"])
    rain = rain.fillna(0)

    """
        학습 데이터 셋에서 사용하는 데이터만 rain_filtered로 추출
    """
    # 침수 발생 시간대
    flood = pd.read_parquet("data/final/gangnam_final_flood_grid.parquet")
    flood["SAT_DATE"] = pd.to_datetime(flood["SAT_DATE"])
    flood["END_DATE"] = pd.to_datetime(flood["END_DATE"])
    flood["END_DATE"] = flood["END_DATE"].fillna(
        flood["SAT_DATE"].dt.normalize() + pd.Timedelta(days=1)
    )

    flood_hours = set()
    for _, row in flood[flood["IS_FLOODDED"] == 1].itertuples(index=False):
        hours = pd.date_range(row.SAT_DATE.floor("h"), row.END_DATE.floor("h"), freq="h")
        flood_hours.update(hours)

    # 강수 0 초과 시간대
    rain_hours = set(rain[rain["rain_1h"] > 0]["time"].unique())

    # 강수 침수 합
    valid_times = flood_hours | rain_hours
    rain_filtered = rain[rain["time"].isin(valid_times)]

    print(f"전체 시간대:    {rain['time'].nunique()}")
    print(f"필터링 후:      {rain_filtered['time'].nunique()}")
    """
        추출 끝
    """

    grid_points = grid[["lon","lat"]].values.astype(np.float64)

    os.makedirs("data/tmp_parquet", exist_ok=True)

    for i, (t, rain_now) in enumerate(rain_filtered.groupby("time")):
 
        print(f"{i}번째 시간 처리중: {t}")

        rain_now = rain_now[rain_now["station"].isin(station_coords)]
        if len(rain_now) == 0:
            print(f"  ->{t}: 유효한 station 없음, 스킵")
            continue

        rain_now = rain_now.sort_values("station")

        station_points = np.array([
            station_coords[s] for s in rain_now["station"]
        ], dtype=np.float64)

        temp = pd.DataFrame({
            "grid_id": grid["grid_id"],
            "time": t
        })

        for feature in features:

            station_values = rain_now[feature].values

            temp[feature] = idw(
                grid_points,
                station_points,
                station_values
            ).astype(np.float32)

        temp.to_parquet(
            f"data/tmp_parquet/grid_{i}.parquet",
            engine="pyarrow",
            compression="snappy"
        )

    print("parquet 파일 생성 완료")

        
if __name__ == "__main__":
    generate_grid_rain_parquet() 