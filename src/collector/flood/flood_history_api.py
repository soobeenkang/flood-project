import geopandas as gpd
import pandas as pd
import numpy as np
import os

def geo_flood():
    

    flood_gdf = gpd.read_file('data/flood/gangnam_flood_data.geojson')
    grid_gdf = gpd.read_file('data/grid/gangnam_grid.geojson')

    if grid_gdf.crs != flood_gdf.crs:
        print("불일치")
        flood_gdf = flood_gdf.to_crs(grid_gdf.crs)

    #공간 결합
    joined= gpd.sjoin(
        grid_gdf, 
        flood_gdf,
        how='left',
        predicate='intersects'
    )

    #grid_id별로 그룹화해서 침수발생연도개수(FLOOD_FREQ)+가장깊은수심(DEPTH) 통계냄
    grid_agg = joined.groupby('grid_id').agg({
        'F_YR':lambda x: x.nunique(),
        'DEPTH': 'max'
    }).reset_index()

    # 컬럼명 정리
    grid_agg.columns=['grid_id','FLOOD_FREQ','MAX_DEPTH']

    grid_agg['FLOOD_FREQ']=grid_agg['FLOOD_FREQ'].fillna(0).astype(int)

    final_grid = grid_gdf.merge(grid_agg,on='grid_id')
    print("데이터 붙임 완료")
    os.makedirs("data/grid", exist_ok=True)

    output_path = "data/flood/gangnam_final_flood_grid.geojson"

    final_grid.to_file(output_path, driver="GeoJSON")
    # .to_file('gangnam_final_flood_grid.geojson',driver='GeoJSON')

    # 데이터 분석용 csv (도형 정보 제외)
    # final_grid.drop(columns='geometry').to_csv('gangnam_final_flood_grid.csv',index=False,encoding='utf-8-sig')

if __name__ =="__main__":
    geo_flood()