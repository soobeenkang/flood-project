import geopandas as gpd
import pandas as pd
import os

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
            gangnam_gdf=gdf[gdf['ADDRESS'].str.contains('강남구',na=False)].copy()
            all_data.append(gangnam_gdf)

    print("데이터 추출 완료")        

    total_flood_gdf = pd.concat(all_data, ignore_index=True)
    total_flood_gdf=total_flood_gdf.drop(columns='피해위치(')
    # print(total_flood_gdf.columns)
    os.makedirs("data/flood", exist_ok=True)
    output_path = "data/flood/gangnam_flood_data.geojson"
    total_flood_gdf.to_file(output_path, driver="GeoJSON")

    print("파일 생성 완료")

if __name__ == "__main__":
    gangnam_flood_data()