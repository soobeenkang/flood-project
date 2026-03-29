"""
    그리드 셀과 시간 당 강수량을
    idw 함수를 사용해 매핑하는 코드
"""
import geopandas as gpd
import pandas as pd
import numpy as np
from numba import njit
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
import os

# 가까운 관측소 5개만 사용
K_NEIGHBORS = 5
R = 10000 # 10km

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

# --- k개의 관측소와의 거리를 이용해 가중치로 해당 그리드 셀 강수 계산 ---
@njit
def idw_k(grid_points, station_points, station_values):

    num = 0.0
    den = 0.0

    for j in range(station_points.shape[0]):
        dx = grid_points[0] - station_points[j, 0]
        dy = grid_points[1] - station_points[j, 1]
        dist = (dx*dx + dy*dy) ** 0.5

        if dist == 0:
            dist = 1e-6

        w = 1.0 / (dist*dist)
        num += w * station_values[j]
        den += w

    return num / den


# --- 관측소 좌표로 KDTree 만들기 ---
def build_kdtree(station_coords_dict):
    station_ids = np.array(list(station_coords_dict.keys()))
    station_locs = np.array(list(station_coords_dict.values()))
    tree = cKDTree(station_locs)
    return tree, station_ids, station_locs


# --- 이웃 계산 ---
def precompute_neighbors(grid_points, tree):
    dist, indices = tree.query(
        grid_points,
        k=K_NEIGHBORS,
        distance_upper_bound=R
    )
    return dist, indices


# --- 시간 당 계산 ---
def process_one(t, rain_now, grid_points, neighbor_dist, neighbor_idx,
                station_ids, station_locs, grid_ids, out_dir, idx):
    
    station_to_idx = {s: i for i, s in enumerate(station_ids)}
    
    temp = pd.DataFrame({"grid_id": grid_ids, "time": t})

    for feature in features:

        values = np.full(len(station_ids), np.nan, dtype=np.float64)

        for s, v in zip(rain_now["station"], rain_now[feature]):
            if s in station_to_idx:
                values[station_to_idx[s]] = v

        result = np.zeros(len(grid_points), dtype=np.float32)

        for i, gp in enumerate(grid_points):

            nbrs = neighbor_idx[i]
            dists = neighbor_dist[i]

            valid = np.isfinite(dists) & (nbrs < len(station_ids))

            if not np.any(valid):
                result[i] = 0.0
                continue

            nbrs = nbrs[valid]

            vals = values[nbrs]
            mask = ~np.isnan(vals)

            if not np.any(mask):
                result[i] = 0.0
                continue

            valid_locs = station_locs[nbrs][mask]
            valid_vals = vals[mask]

            result[i] = idw_k(gp, valid_locs, valid_vals)

        temp[feature] = result
    
    temp.to_parquet(
        os.path.join(out_dir, f"grid_{idx}.parquet"),
        engine="pyarrow",
        compression="snappy"
    )


# 최종 함수
def generate_grid_rain_parquet():
    #   서울 그리드 데이터 불러오기
    grid = gpd.read_file("data/grid/seoul_grid.geojson")

    grid = grid.to_crs(epsg=5186)

    if "grid_id" not in grid.columns:
        grid = grid.reset_index().rename(columns={"index": "grid_id"})

    grid_points = np.array(list(zip(grid.geometry.x, grid.geometry.y)))
    grid_ids = grid["grid_id"].values

    #   aws 강수 피쳐 불러오기
    rain = pd.read_csv("data/prev_rainfall/aws_rainfall_features.csv")
    rain["time"] = pd.to_datetime(rain["time"])
    rain["station"] = rain["station"].astype("int32")
    rain = rain.fillna(0)

    # station meta data 불러오기
    station_meta = pd.read_csv("data/raw_aws/station_coords.csv")
    station_meta = station_meta[station_meta["expired_date"].isna()].drop_duplicates(subset="station")
    # 좌표계 맞추기
    station_gdf = gpd.GeoDataFrame(
        station_meta,
        geometry=gpd.points_from_xy(station_meta["lon"], station_meta["lat"]),
        crs="EPSG:4326"
    ).to_crs(epsg=5186)

    station_coords_dict = dict(
        zip(station_gdf["station"],
            zip(station_gdf.geometry.x, station_gdf.geometry.y))
    )

    # 실제 rain에 있는 관측소만 사용
    valid_stations = set(rain["station"].unique())
    station_coords_dict = {k: v for k, v in station_coords_dict.items()
                           if k in valid_stations}
    print(f"사용 관측소 수: {len(station_coords_dict)}")

    # KDTree 생성 & 이웃 계산
    tree, station_ids, station_locs = build_kdtree(station_coords_dict)
    neighbor_dist, neighbor_idx = precompute_neighbors(grid_points, tree)

    """
        학습 데이터 셋에서 사용하는 데이터만 rain_filtered로 추출
    """
    # 침수 발생 시간대 필터링
    flood = pd.read_parquet("data/final/seoul_final_flood_grid.parquet")
    flood_hours = set()
    for row in flood.itertuples(index=False):
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

    # 최종 tmp parquet 들을 병렬로 계산
    os.makedirs("data/tmp_parquet", exist_ok=True)

    groups = list(rain_filtered.groupby("time"))

    Parallel(n_jobs=4, verbose=10)(
        delayed(process_one)(
            t, rain_now, grid_points, neighbor_dist, neighbor_idx,
            station_ids, station_locs, grid_ids, "data/tmp_parquet", i
        )
        for i, (t, rain_now) in enumerate(groups)
    )

    print("parquet 파일 생성 완료")

        
if __name__ == "__main__":
    generate_grid_rain_parquet() 