import geopandas as gpd
import pandas as pd
import os
from datetime import datetime, timedelta

def gangnam_flood_data():
    raw_data_path = "data/year/" 
    all_data = []

    for year in range(2010, 2026):
        file = f"{raw_data_path}/{year}/서울시_{year}.shp"
        if os.path.exists(file):
            gdf = gpd.read_file(file, encoding='cp949').to_crs(epsg=4326)
            gdf=gdf.rename(columns={
                'F_ZONE_NM': 'ADDRESS',
                'F_AREA': 'AREA',
                'F_SHIM': 'DEPTH'
            })
            all_data.append(gdf)

        
    print("데이터 추출 완료")

    total_flood_gdf = pd.concat(all_data, ignore_index=True)
    total_flood_gdf = total_flood_gdf.dropna(subset=['F_SAT_YMD', 'F_END_YMD'])

    dates = pd.to_datetime(total_flood_gdf['F_SAT_YMD'], format='mixed',errors='coerce')
    hours = pd.to_timedelta(total_flood_gdf['F_SAT_TM'].astype(int), unit='h')
    total_flood_gdf['F_SAT_YMD']= dates+hours

    dates = pd.to_datetime(total_flood_gdf['F_END_YMD'],format='mixed',errors='coerce')
    hours = pd.to_timedelta(total_flood_gdf['F_END_TM'].astype(int), unit='h')
    total_flood_gdf['F_END_YMD']= dates+hours

    total_flood_gdf = total_flood_gdf.dropna(subset=['F_SAT_YMD', 'F_END_YMD'])
    total_flood_gdf = total_flood_gdf.reset_index(drop=True)

    total_flood_gdf = total_flood_gdf.dropna(axis=1, how='all')
    total_flood_gdf = total_flood_gdf.drop(columns=['F_SAT_TM', 'F_END_TM'])

    os.makedirs("data/flood", exist_ok=True)
    output_path = "data/flood/seoul_flood_data.geojson"
    total_flood_gdf.to_file(output_path, driver="GeoJSON")

    print("파일 생성 완료")

if __name__ == "__main__":
    gangnam_flood_data()