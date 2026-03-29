"""
    그리드 셀마다 이웃 관측소를 찾고
    시간 당 강수량을 idw 함수를 사용해 매핑하는 코드
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


#   k개의 관측소와의 거리를 이용해 가중치로 해당 그리드 셀 강수 계산 함수
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


#   관측소 좌표로 KDTree 만들기 함수
def build_kdtree(station_coords_dict):
    station_ids = np.array(list(station_coords_dict.keys()))
    station_locs = np.array(list(station_coords_dict.values()))
    tree = cKDTree(station_locs)
    return tree, station_ids, station_locs


#   좌표 변경 시점마다 KDTree 만들기 함수
def build_period_trees(station_meta, rain_times, valid_stations, grid_points):
    station_meta = station_meta[station_meta["station"].isin(valid_stations)].copy()
    station_meta["start_date"] = pd.to_datetime(station_meta["start_date"])
    station_meta["expired_date"] = pd.to_datetime(station_meta["expired_date"])

    # 좌표 변경 시점
    breakpoints = sorted(set(
        station_meta["start_date"].dropna().tolist() +
        station_meta["expired_date"].dropna().tolist()
    ))

    rain_times = sorted(rain_times)

    # bp 시점마다 구간별 트리 만들기
    period_trees = []
    for bp in breakpoints:
        active = station_meta[
            (station_meta["start_date"] <= bp) &
            (station_meta["expired_date"].isna() | (station_meta["expired_date"] > bp))
            ].drop_duplicates(subset="station")
        
        if len(active) == 0:
            continue

        # 좌표 변환
        active_gdf = gpd.GeoDataFrame(
            active,
            geometry=gpd.points_from_xy(active["lon"], active["lat"]),
            crs="EPSG:4326"
        ).to_crs(epsg=5186)

        coords_dict = dict(
            zip(active_gdf["station"],
                zip(active_gdf.geometry.x, active_gdf.geometry.y))
        )

        tree, station_ids, station_locs = build_kdtree(coords_dict)
        neighbor_dist, neighbor_idx = precompute_neighbors(grid_points, tree)

        period_trees.append((bp, tree, station_ids, station_locs, neighbor_dist, neighbor_idx))

    return period_trees


#   t에 해당하는 tree return 함수
def get_tree_for_time(t, period_trees):
    result = None
    for bp, tree, station_ids, station_locs, neighbor_dist, neighbor_idx in period_trees:
        if bp <= t:
            result = (tree, station_ids, station_locs, neighbor_dist, neighbor_idx)
        else:
            break
    return result


#   이웃 계산 with max distance 함수
def precompute_neighbors(grid_points, tree):
    dist, indices = tree.query(
        grid_points,
        k=K_NEIGHBORS,
        distance_upper_bound=R
    )
    return dist, indices


#   시간 당 계산 함수
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
        os.path.join(out_dir, f"aws_tmp_{idx}.parquet"),
        engine="pyarrow",
        compression="snappy"
    )


#   최종 함수
def generate_grid_rain_parquet():
    # --- 서울 그리드 데이터 불러오기 ---
    grid = gpd.read_file("data/grid/seoul_grid.geojson")

    if "grid_id" not in grid.columns:
        grid = grid.reset_index().rename(columns={"index": "grid_id"})

    grid_points = np.array(list(zip(grid.geometry.x, grid.geometry.y)))
    grid_ids = grid["grid_id"].values

    # --- aws 강수 피쳐 불러오기 ---
    rain = pd.read_csv("data/rainfall_history/aws_rainfall_features.csv")
    rain["time"] = pd.to_datetime(rain["time"])
    rain["station"] = rain["station"].astype("int32")
    rain = rain.fillna(0)

    # --- station meta data 불러오기 ---
    station_meta = pd.read_csv("data/rainfall_history/station_coords.csv")
    valid_stations = set(rain["station"].unique())

    # --- 좌표 변경 시점 KDTree 미리 만들기 ---
    rain_times = rain["time"].unique()
    period_trees = build_period_trees(station_meta, rain_times, valid_stations, grid_points)
    print(f"KDTree 구간 수: {len(period_trees)}") 

    """
        학습 데이터 셋에서 사용하는 데이터만 rain_filtered로 추출
    """
    #   침수 발생 시간대 필터링
    flood = pd.read_parquet("data/final/seoul_final_flood_grid.parquet")
    flood_hours = set()
    for row in flood.itertuples(index=False):
        hours = pd.date_range(row.SAT_DATE.floor("h"), row.END_DATE.floor("h"), freq="h")
        flood_hours.update(hours)

    #   강수 0 초과 시간대
    rain_hours = set(rain[rain["rain_1h"] > 0]["time"].unique())

    #   강수 침수 합
    valid_times = flood_hours | rain_hours
    rain_filtered = rain[rain["time"].isin(valid_times)]

    print(f"전체 시간대:    {rain['time'].nunique()}")
    print(f"필터링 후:      {rain_filtered['time'].nunique()}")
    """
        추출 끝
    """

    # --- 최종 tmp parquet 들을 순차 계산 ---
    os.makedirs("data/tmp_parquet", exist_ok=True)

    for i, (t, rain_now) in enumerate(rain_filtered.groupby("time")):
        print(f"{i}번째 시간 처리중: {t}")

        tree_info = get_tree_for_time(t, period_trees)
        if tree_info is None:
            print(f"    {t}: 해당 구간 트리 없음")
            continue

        _, station_ids, station_locs, neighbor_dist, neighbor_idx = tree_info

        process_one(
            t, rain_now, grid_points, neighbor_dist, neighbor_idx,
            station_ids, station_locs, grid_ids, "data/tmp_parquet", i
        )

    print("parquet 파일 생성 완료")

        
if __name__ == "__main__":
    generate_grid_rain_parquet() 