import geopandas as gpd
import numpy as np
from shapely.geometry import box
import os


def generate_seoul_grid():

    shp_path = "data/boundary/BND_SIGUNGU_PG.shp"

    print("시군구 데이터 로드 중...")
    gdf = gpd.read_file(shp_path)
    
    # 서울시 추출
    seoul = gdf[gdf["SIGUNGU_CD"].astype(str).str.startswith("11")]

    print("서울시 추출 완료")

    # 미터 좌표계로 변환 for generate grid
    seoul = seoul.to_crs(epsg=5179)

    seoul_union = seoul.geometry.unary_union

    minx, miny, maxx, maxy = seoul.total_bounds
    
    # 100m
    grid_size = 100

    grid_cells = []

    print("grid 생성 중...")

    for x in np.arange(minx, maxx, grid_size):
        for y in np.arange(miny, maxy, grid_size):

            cell = box(x, y, x + grid_size, y + grid_size)

            if cell.intersects(seoul_union):
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

    output_path = "data/grid/seoul_grid.geojson"

    grid.to_file(output_path, driver="GeoJSON")

    print("저장 완료:", output_path)

    return grid


if __name__ == "__main__":
    generate_seoul_grid()