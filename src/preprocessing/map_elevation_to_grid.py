import geopandas as gpd
import pandas as pd
import numpy as np
import os


def map_elevation_to_existing_gangnam_grid():
    grid_path = "data/grid/gangnam_grid.geojson"
    contour_path = "contour_points.csv"
    output_path = "data/grid/gangnam_grid_elevation.parquet"

    # 기존 강남구 grid 로드
    print("기존 강남구 grid 로드 중...")
    grid = gpd.read_file(grid_path)

    # 혹시 grid_id가 없으면 생성
    if "grid_id" not in grid.columns:
        grid["grid_id"] = range(len(grid))

    # 좌표계 통일
    # 기존 코드 기준으로 작업 좌표계는 EPSG:5179 사용
    if grid.crs is None:
        raise ValueError("gangnam_grid.geojson에 CRS 정보가 없습니다.")
    grid = grid.to_crs(epsg=5179)

    print("grid 개수:", len(grid))

    # 등고선 포인트 로드
    print("등고선 포인트 로드 중...")
    df_points = pd.read_csv(contour_path)

    points_gdf = gpd.GeoDataFrame(
        df_points,
        geometry=gpd.points_from_xy(df_points["x"], df_points["y"]),
        crs="EPSG:5186"   # 수치지형도 데이터 좌표계
    )

    # 좌표계 통일
    points_gdf = points_gdf.to_crs(epsg=5179)

    # spatial join
    print("공간 매핑 중...")
    joined = gpd.sjoin(
        points_gdf,
        grid[["grid_id", "geometry"]],
        how="inner",
        predicate="intersects"
    )

    print("매핑된 포인트 수:", len(joined))

    # grid별 평균 elevation 계산
    elevation_per_grid = (
        joined.groupby("grid_id")["elevation"]
        .mean()
        .reset_index()
    )

    # grid_id 기준으로 elevation 병합
    result = grid[["grid_id"]].merge(elevation_per_grid, on="grid_id", how="left")

    # 결측값 처리
    print("결측값 처리 중...")
    overall_mean = result["elevation"].mean()
    result["elevation"] = result["elevation"].fillna(overall_mean)

    # float32로 변환
    result["elevation"] = result["elevation"].astype(np.float32)

    # 최종적으로 grid_id, elevation만 저장
    result = result[["grid_id", "elevation"]]

    # 저장
    os.makedirs("data/grid", exist_ok=True)
    result.to_parquet(output_path, index=False)

    print("완료:", output_path)
    print(result.dtypes)

    return result


if __name__ == "__main__":
    map_elevation_to_existing_gangnam_grid()