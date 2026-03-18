import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import box
import os


def generate_gangnam_grid_with_elevation():

    
    # 강남구 그리드 생성
    shp_path = "src/collector/data/boundary/BND_SIGUNGU_PG.shp"

    print("시군구 데이터 로드 중...")
    gdf = gpd.read_file(shp_path)

    gangnam = gdf[gdf["SIGUNGU_NM"] == "강남구"]
    gangnam = gangnam.to_crs(epsg=5179)

    gangnam_union = gangnam.geometry.union_all()
    minx, miny, maxx, maxy = gangnam.total_bounds

    grid_size = 100
    grid_cells = []

    print("grid 생성 중...")

    for x in np.arange(minx, maxx, grid_size):
        for y in np.arange(miny, maxy, grid_size):
            cell = box(x, y, x + grid_size, y + grid_size)

            if cell.intersects(gangnam_union):
                grid_cells.append(cell)

    grid = gpd.GeoDataFrame(geometry=grid_cells, crs="EPSG:5179")
    grid["grid_id"] = range(len(grid))

    print("grid 개수:", len(grid))

    # 등고선 포인트 로드
    print("등고선 포인트 로드 중...")
    df_points = pd.read_csv("contour_points.csv")

    points_gdf = gpd.GeoDataFrame(
        df_points,
        geometry=gpd.points_from_xy(df_points["x"], df_points["y"]),
        crs="EPSG:5186"   #수치지형도 데이터 좌표계
    )

    #좌표계 통일
    points_gdf = points_gdf.to_crs(epsg=5179)
   
    #spatial join
    print("공간 매핑 중...")

    joined = gpd.sjoin(
        points_gdf,
        grid,
        how="inner",
        predicate="intersects"   
    )

    print("매핑된 포인트 수:", len(joined))

    # grid별 평균값 계산
    # 나중에 보수적으로 바꾸고싶다면 낮은 값 넣기로 변경 가능
    elevation_per_grid = (
        joined.groupby("grid_id")["elevation"]
        .mean()
        .reset_index()
    )

    # grid에 고도 정보 합치기
    grid = grid.merge(elevation_per_grid, on="grid_id", how="left")

    # 결측값 처리
    print("결측값 처리 중...")
    overall_mean = grid["elevation"].mean()
    grid["elevation"] = grid["elevation"].fillna(overall_mean)

    # centroid 계산 및 위경도 변환
    print("위경도 변환 중...")

    centroids = grid.centroid

    # geometry 변환
    grid = grid.to_crs(epsg=4326)

    print("\n=== 포인트 범위 ===")
    print(points_gdf.total_bounds)

    print("\n=== grid 범위 ===")
    print(grid.total_bounds)

    # centroid도 따로 변환
    centroids = gpd.GeoSeries(centroids, crs="EPSG:5179").to_crs(epsg=4326)

    grid["lon"] = centroids.x
    grid["lat"] = centroids.y

    # 저장
    os.makedirs("data/grid", exist_ok=True)
    print(points_gdf.total_bounds)
    print(grid.total_bounds)
    print("매핑 개수:", len(joined))
    output_path = "data/grid/gangnam_grid_with_elevation.geojson"
    grid.to_file(output_path, driver="GeoJSON")

    print("완료:", output_path)

    return grid


if __name__ == "__main__":
    generate_gangnam_grid_with_elevation()
