import geopandas as gpd
import numpy as np
from shapely.geometry import box
import os


def generate_gangnam_grid():

    shp_path = "data/boundary/BND_SIGUNGU_PG.shp"

    print("시군구 데이터 로드 중...")
    gdf = gpd.read_file(shp_path)
    
    # 강남구 추출
    gangnam = gdf[gdf["SIGUNGU_NM"] == "강남구"]

    print("강남구 추출 완료")

    # 미터 좌표계로 변환 for generate grid
    gangnam = gangnam.to_crs(epsg=5179)

    gangnam_union = gangnam.geometry.unary_union

    minx, miny, maxx, maxy = gangnam.total_bounds
    
    # 100m
    grid_size = 100

    grid_cells = []

    print("grid 생성 중...")

    for x in np.arange(minx, maxx, grid_size):
        for y in np.arange(miny, maxy, grid_size):

            cell = box(x, y, x + grid_size, y + grid_size)

            if cell.intersects(gangnam_union):
                grid_cells.append(cell)

    grid = gpd.GeoDataFrame(geometry=grid_cells, crs="EPSG:5179")

    # calculate centroid in meter
    centroids = grid.centroid

    # 위경도로 변환
    grid = grid.to_crs(epsg=4326)

    # centroid도 위경도로 변환
    centroids = gpd.GeoSeries(centroids, crs="EPSG:5179").to_crs(epsg=4326)

    grid["lon"] = centroids.x
    grid["lat"] = centroids.y
    
    # grid id
    grid["grid_id"] = range(len(grid))

    print("grid 개수:", len(grid))

    os.makedirs("data/grid", exist_ok=True)

    output_path = "data/grid/gangnam_grid.geojson"

    grid.to_file(output_path, driver="GeoJSON")

    print("저장 완료:", output_path)

    return grid


if __name__ == "__main__":
    generate_gangnam_grid()