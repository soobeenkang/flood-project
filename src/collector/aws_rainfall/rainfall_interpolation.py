"""
    필터링: 침수 이력이 있는 그리드, 침수 발생 시간 + 강수가 있는 시간
    그리드 셀마다 이웃 관측소를 찾고
    시간 당 강수량을 idw 함수를 사용해 매핑하는 코드
"""
import geopandas as gpd
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
import os

# 가까운 관측소 5개만 사용
K_NEIGHBORS = 5
R = 10000 # 10km

# 강수 feature 7개
features = [
    "rain_1h", "rain_3h", "rain_6h", "rain_12h", 
    "rain_24h", "rain_intensity", "rain_max_3h"
]


# --- 모든 feature를 한 번에 백터 처리
def compute_idw_all_features(
        rain_values,    # (F, S)
        neighbor_idx,   # (G, K)
        neighbor_dist   # (G, K)
):
    G, K = neighbor_idx.shape
    F, S = rain_values.shape

    # 유효 이웃 필터링 마스크 G, K
    valid_idx_mask = neighbor_idx < S
    finite_mask = np.isfinite(neighbor_dist)
    mask = valid_idx_mask & finite_mask     # G, K

    # 가중치 계산 G, K - 유효하지 않으면 0
    weights = np.where(mask, 1.0 / (neighbor_dist ** 2 + 1e-12), 0.0)

    # 안전한 인덱스 G, K - 범위 초과는 0으로
    safe_idx = np.where(valid_idx_mask, neighbor_idx, 0)

    # 관측소 값 모으기 F, S -> F, G, K
    vals = rain_values[:, safe_idx]

    # NaN 처리 마스크 F, G, K
    nan_mask = np.isnan(vals)
    combined = mask[np.newaxis, :, :] & ~nan_mask

    w = np.where(combined, weights[np.newaxis, :, :], 0.0) # F, G, K
    v = np.nan_to_num(vals)

    num = np.sum(w * v, axis=2) # F, G
    den = np.sum(w, axis=2)

    result = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)

    return result.T.astype(np.float32) # G, F


#   관측소 좌표로 KDTree 만들기 함수
def build_kdtree(station_coords_dict):
    station_ids = np.array(list(station_coords_dict.keys()))
    station_locs = np.array(list(station_coords_dict.values()))
    tree = cKDTree(station_locs)
    return tree, station_ids, station_locs


#   이웃 계산 with max distance 함수
def precompute_neighbors(grid_points, tree):
    dist, indices = tree.query(
        grid_points,
        k=K_NEIGHBORS,
        distance_upper_bound=R
    )
    return dist, indices


#   좌표 변경 시점마다 KDTree 만들기 함수
def build_period_trees(station_meta, valid_stations, grid_points):
    station_meta = station_meta[station_meta["station"].isin(valid_stations)].copy()
    station_meta["start_date"] = pd.to_datetime(station_meta["start_date"])
    station_meta["expired_date"] = pd.to_datetime(station_meta["expired_date"])

    # 좌표 변경 시점
    breakpoints = sorted(set(
        station_meta["start_date"].dropna().tolist() +
        station_meta["expired_date"].dropna().tolist()
    ))

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
            zip(active_gdf["station"].values,
                zip(active_gdf.geometry.x.values, active_gdf.geometry.y.values))
        )

        tree, station_ids, station_locs = build_kdtree(coords_dict)
        neighbor_dist, neighbor_idx = precompute_neighbors(grid_points, tree)

        period_trees.append((bp, station_ids, station_locs, neighbor_dist, neighbor_idx))

    return period_trees


#   t에 해당하는 tree return 함수
def get_tree_for_time(t, period_trees):
    result = None
    for bp, station_ids, station_locs, neighbor_dist, neighbor_idx in period_trees:
        if bp <= t:
            result = (station_ids, station_locs, neighbor_dist, neighbor_idx)
        else:
            break
    return result


#   시간 당 계산 함수
def process_one(i, t, station_ids, station_locs, neighbor_dist, neighbor_idx,
                station_to_idx,
                rain_stations, rain_matrix, grid_ids):
    
    if i % 1000 == 0:
        print(f"{i}번째 시간 처리중: {t}")

    S = len(station_ids)
    F = len(features)
    
    # (F, S) 배열 구성
    values = np.full((F, S), np.nan, dtype=np.float64)
    for k, s in enumerate(rain_stations):
        if s in station_to_idx:
            values[:, station_to_idx[s]] = rain_matrix[:, k]

    # 전체 강수 feature 한번에 idw
    result = compute_idw_all_features(values, neighbor_idx, neighbor_dist)

    temp = pd.DataFrame(result, columns=features)
    temp.insert(0, "grid_id", grid_ids)
    temp.insert(1, "time", t)
    
    return temp


#   최종 함수
def generate_grid_rain_parquet():
    # --- 침수 그리드 데이터 불러오기 ---
    flood = pd.read_parquet("data/final/seoul_final_flood_grid.parquet")
    flood = flood.dropna(subset=["F_SAT_YMD", "F_END_YMD"])

    # --- 서울 그리드 데이터 불러오기 ---
    grid = gpd.read_file("data/grid/seoul_grid.geojson").to_crs(epsg=5186)

    if "grid_id" not in grid.columns:
        grid = grid.reset_index().rename(columns={"index": "grid_id"})

    #   침수 발생 이력 그리드만 필터링
    valid_grids = flood["grid_id"].unique()
    grid = grid[grid["grid_id"].isin(valid_grids)]

    grid_points = np.vstack([grid.geometry.centroid.x.values,
                             grid.geometry.centroid.y.values]).T
    grid_ids = grid["grid_id"].values

    # --- aws 강수 피쳐 불러오기 ---
    rain = pd.read_csv("data/rainfall_history/aws_rainfall_features.csv") 
    rain["time"] = pd.to_datetime(rain["time"])
    rain["station"] = rain["station"].astype("int32")

    # --- station meta data 불러오기 ---
    station_meta = pd.read_csv("data/rainfall_history/station_coords.csv")
    valid_stations = set(rain["station"].unique())

    # --- 좌표 변경 시점 KDTree 미리 만들기 ---
    period_trees = build_period_trees(station_meta, valid_stations, grid_points)
    print(f"KDTree 구간 수: {len(period_trees)}") 

    #   침수 발생 시간대 필터링
    flood_hours = set()
    for row in flood.itertuples(index=False):
        hours = pd.date_range(row.F_SAT_YMD.floor("h"), row.F_END_YMD.floor("h"), freq="h")
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
    os.makedirs("data/tmp_rainfall_parquet", exist_ok=True)

    # chunk 별로 저장
    CHUNK_SIZE = 100
    rain_grouped = rain_filtered.groupby("time")
    times = rain_filtered["time"].unique()

    for chunk_start in range(0, len(times), CHUNK_SIZE):
        chunk_times = times[chunk_start:chunk_start + CHUNK_SIZE]
        tasks = []

        for i, t in enumerate(chunk_times):

            rain_now = rain_grouped.get_group(t)

            tree_info = get_tree_for_time(t, period_trees)
            if tree_info is None:
                print(f"    {t}: 해당 구간 트리 없음")
                continue

            station_ids, station_locs, neighbor_dist, neighbor_idx = tree_info

            station_to_idx = {s: j for j, s in enumerate(station_ids)}
            
            rain_stations = rain_now["station"].values.astype("int32")
            rain_matrix = rain_now[features].values.T.astype(np.float32)
            
            neighbor_dist = neighbor_dist.astype(np.float32)

            tasks.append((
                chunk_start + i, t,
                station_ids, station_locs, neighbor_dist, neighbor_idx,
                station_to_idx,
                rain_stations, rain_matrix
            ))

        # 시간 당 결과 results 에 저장
        results = Parallel(n_jobs=4, backend="threading", verbose=5)(
            delayed(process_one)(
                i, t,
                station_ids, station_locs, neighbor_dist, neighbor_idx,
                station_to_idx,
                rain_stations, rain_matrix,
                grid_ids
            )
            for (i, t,
                station_ids, station_locs, neighbor_dist, neighbor_idx,
                station_to_idx,
                rain_stations, rain_matrix) in tasks
        )

        # 최종 parquet 파일 생성
        df_chunk = pd.concat(results, ignore_index=True)

        df_chunk.to_parquet(
            f"data/tmp_rainfall_parquet/chunk_{chunk_start}.parquet",
            engine="pyarrow",
            compression="snappy"
        )

        print(f"chunk {chunk_start} 완료")

        
if __name__ == "__main__":
    generate_grid_rain_parquet() 