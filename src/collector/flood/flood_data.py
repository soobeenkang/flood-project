import geopandas as gpd
import pandas as pd
import os
import re

def process_flood_dates(row):

    if pd.notnull(row.get('피해일시')) and row['피해일시'] != "":
        start_day = str(row['피해일시'])[:8]
        row['F_SAT_YMD'] = start_day
        print("2025 데이터 처리 완료")
    else:
        sat = str(row['F_SAT_YMD']) if pd.notnull(row['F_SAT_YMD']) else ""
        if "-" in sat:
            start_day = sat.split("-")[0].strip()
            row['F_SAT_YMD'] = start_day
        else:
            start_day = sat

    #F_END_YMD
    if pd.isnull(row['F_END_YMD']) or row['F_END_YMD'] == "":
        disa_nm = row['F_DISA_NM']
        if isinstance(disa_nm, str) and "~" in disa_nm:
            try:
                _, end_part = disa_nm.split("~")
                nums = re.findall(r'\d+', end_part)

                if len(start_day) >= 8: 
                    if len(nums) == 2: 
                        row['F_END_YMD'] = start_day[:4] + nums[0].zfill(2) + nums[1].zfill(2)
                    elif len(nums) == 1:
                        row['F_END_YMD'] = start_day[:6] + nums[0].zfill(2)
            except Exception:
                pass # 파싱 실패 시 빈칸 유지
        else:
            row['F_END_YMD']=start_day

    def format_datetime(d_str, t_val):
        d_str=str(d_str).strip() if pd.notna(d_str) else ""
        if len(d_str)>=8:
            d_str=d_str.replace("-","")
            formatted_date = f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}"

            if pd.notna(t_val) and str(t_val).strip() !="":
                if len(str(t_val))==4:
                    hh=str(t_val)
                    return f"{formatted_date}-{hh}"
                elif len(str(t_val))<=2:
                    hh=str(t_val).zfill(2)
                    return f"{formatted_date}-{hh}00"
            return formatted_date    
    row['SAT_DATE'] = format_datetime(row.get('F_SAT_YMD'), row.get('F_SAT_TM'))
    row['END_DATE'] = format_datetime(row.get('F_END_YMD'), row.get('F_END_TM'))
    return row


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
    total_flood_gdf = total_flood_gdf.apply(process_flood_dates, axis=1)
    total_flood_gdf=total_flood_gdf.drop(columns=['피해위치(','피해일시'])
    # print(total_flood_gdf.columns)
    os.makedirs("data/flood", exist_ok=True)
    output_path = "data/flood/gangnam_flood_data.geojson"
    total_flood_gdf.to_file(output_path, driver="GeoJSON")

    print("파일 생성 완료")

if __name__ == "__main__":
    gangnam_flood_data()