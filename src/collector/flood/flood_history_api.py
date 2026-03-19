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
        flood_gdf,
        grid_gdf[['grid_id','geometry']],
        how='inner',
        predicate='intersects'
    )

    #grid_id별로 그룹화해서 침수발생연도개수(FLOOD_FREQ)+가장깊은수심(DEPTH) 통계냄
    print("날짜 및 그리드별 데이터 집계 중")
    grid_history = joined.groupby('grid_id').agg({
        'SAT_DATE': lambda x: ', '.join(x.dropna().astype(str)),
        'END_DATE': lambda x: ', '.join(x.dropna().astype(str)),
        'DEPTH': lambda x: ', '.join(x.dropna().astype(str)),
        'F_YR':'count'
    }).reset_index()

    # 컬럼명 정리
    grid_history.rename(columns={
        'DEPTH': 'FLOOD_DEPTH_LIST',
        'F_YR': 'FLOOD_COUNT'
    }, inplace=True)
    grid_history['IS_FLOODED']=1

    final_grid = grid_gdf.merge(grid_history,on='grid_id',how='left')

    final_grid['FLOOD_COUNT'] = final_grid['FLOOD_COUNT'].fillna(0).astype(int)
    final_grid['IS_FLOODED'] = final_grid['IS_FLOODED'].fillna(0).astype(int)
    final_grid['FLOOD_DEPTH_LIST'] = final_grid['FLOOD_DEPTH_LIST'].fillna("0.0")
    print("데이터 붙임 완료")
    
    final_grid['SAT_DATE'] = final_grid['SAT_DATE'].fillna("")
    final_grid['END_DATE'] = final_grid['END_DATE'].fillna("")

    os.makedirs("data/grid", exist_ok=True)

    output_path = "data/flood/gangnam_final_flood_grid.geojson"

    final_grid.to_file(output_path, driver="GeoJSON")
    # .to_file('gangnam_final_flood_grid.geojson',driver='GeoJSON')

    # 데이터 분석용 csv (도형 정보 제외)
    # final_grid.drop(columns='geometry').to_csv('gangnam_final_flood_grid.csv',index=False,encoding='utf-8-sig')

if __name__ =="__main__":
    geo_flood()