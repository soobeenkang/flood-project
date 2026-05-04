from pathlib import Path
import pandas as pd
import geopandas as gpd


def map_sensor_to_grid():
    # 프로젝트 루트 기준 경로
    base_dir = Path(__file__).resolve().parents[3]

    sensor_csv_path = base_dir / "data" / "sensor" / "sensor_locations.csv"
    grid_geojson_path = base_dir / "data" / "grid" / "seoul_grid.geojson"
    output_csv_path = base_dir / "data" / "grid" / "sensor_grid_map.csv"

    #센서 위치 csv 읽기
    sensor_df = pd.read_csv(sensor_csv_path)

    #센서를 point로 변환
    sensor_gdf = gpd.GeoDataFrame(
        sensor_df,
        geometry=gpd.points_from_xy(sensor_df["lon"], sensor_df["lat"]),
        crs="EPSG:4326"
    )

    #grid geojson 읽기
    grid_gdf = gpd.read_file(grid_geojson_path)

    #좌표계가 다르면 맞춰주기
    if sensor_gdf.crs != grid_gdf.crs:
        sensor_gdf = sensor_gdf.to_crs(grid_gdf.crs)

    #센서가 어느 grid 안에 있는지 찾기
    mapped = gpd.sjoin(
        sensor_gdf,
        grid_gdf[["grid_id", "geometry"]],
        how="left",
        predicate="intersects"
    )

    #필요한 컬럼만 저장
    result = mapped[["sensor_id", "lat", "lon", "grid_id"]].copy()

    #결과 저장
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

    print("센서-grid 매핑 완료")
    print(result)
    print(f"저장 경로: {output_csv_path}")

    print("전체 센서 수:", len(result))
    print("grid_id 없는 센서 수:", result["grid_id"].isna().sum())
    print(result[result["grid_id"].isna()])

    return result


if __name__ == "__main__":
    map_sensor_to_grid()
    