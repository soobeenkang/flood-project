import geopandas as gpd
import pandas as pd
import numpy as np
import os

def geo_flood():
    

    flood_gdf = gpd.read_file('data/flood/seoul_flood_data.geojson')
    grid_gdf = gpd.read_file('data/grid/seoul_grid.geojson')

    if grid_gdf.crs != flood_gdf.crs:
        print("불일치")
        flood_gdf = flood_gdf.to_crs(grid_gdf.crs)

    # 공간 결합 침수 위치가 어떤 grid에 포함되는지
    joined= gpd.sjoin(
        flood_gdf,
        grid_gdf[['grid_id','geometry']],
        how='inner', 
        predicate='intersects' 
    )
    
    joined_dedup = joined.drop_duplicates(subset=['grid_id', 'F_SAT_YMD'])
    print("그리드ID별로 시작,끝 같은 중복값 제거 완료")
    joined_dedup['IS_FLOODED'] = 1
    # grid_id별로 그룹화해서 침수발생연도개수(FLOOD_FREQ)+가장깊은수심(DEPTH) 통계냄
    grid_history = joined_dedup[['grid_id', 'F_SAT_YMD', 'F_END_YMD', 'IS_FLOODED']]
    print("시작, 끝 날짜/시간 및 침수 여부 데이터 추출 완료")
    final_grid = grid_gdf.merge(grid_history, on='grid_id', how='left')

    # 침수 이력이 없는(Null) 그리드들의 결측치 처리
    final_grid['IS_FLOODED'] = final_grid['IS_FLOODED'].fillna(0).astype(int)
    
    print("데이터 병합 완료")

    os.makedirs("data/grid", exist_ok=True)

    # output_path = "data/flood/seoul_final_flood_grid.geojson"
    output_path = "data/flood/seoul_final_flood_grid.parquet"
    final_grid.to_parquet(output_path, engine='pyarrow')
    # final_grid.to_file(output_path, driver="GeoJSON")

if __name__ =="__main__":
    geo_flood()