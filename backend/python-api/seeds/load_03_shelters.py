"""
shelter 테이블에 데이터 올리기

소스 데이터:
    - data/seoul_shleters.xlsx
"""
import os
import pandas as pd
from sqlalchemy import text
from db import get_engine

def preprocess_shelter_type(raw_type):
    """
    대피소 유형(GB_ACMD) 전처리:
    결측치 처리, '공공시설' 정리
    """
    if pd.isna(raw_type):
        return None
    
    type_str = str(raw_type).strip()
    
    if type_str.startswith("공공시설"):
        return "공공시설"
        
    return type_str

def load_shelters(file_path):
    df = pd.read_excel(file_path)
    
    engine = get_engine()
    
    insert_query = text("""
        INSERT INTO shelter (name, address, type, geom)
        VALUES (:name, :address, :type, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326));
    """)
    
    success_count = 0
    print("데이터 삽입을 시작합니다...")
    
    with engine.begin() as connection:
        for index, row in df.iterrows():
            try:
                name = row.get('EQUP_NM')
                address = row.get('LOC_SFPR_A')
                lon = row.get('XCORD') # 경도
                lat = row.get('YCORD') # 위도
                
                # 필수 데이터(이름, 좌표)가 없으면 패스
                if pd.isna(lon) or pd.isna(lat) or pd.isna(name):
                    print(f"[{index}번 행] 필수 데이터(이름 또는 좌표) 누락으로 건너뜁니다.")
                    continue
                
                raw_type = row.get('GB_ACMD')
                shelter_type = preprocess_shelter_type(raw_type)
                
                connection.execute(insert_query, {
                    "name": str(name).strip(),
                    "address": str(address).strip() if pd.notna(address) else None,
                    "type": shelter_type,
                    "lon": float(lon),
                    "lat": float(lat)
                })
                success_count += 1
                
            except Exception as e:
                print(f"[{index}번 행] 에러 발생: {e}")
                # 건너뛰고 계속 진행
                continue

    print(f"--- 작업 완료 ---")
    print(f"총 {success_count}개의 대피소 데이터가 성공적으로 저장되었습니다.")

if __name__ == "__main__":
    FILE_PATH = "data/seoul_shelters.xlsx" 
    
    if os.path.exists(FILE_PATH):
        load_shelters(FILE_PATH)
    else:
        print(f"파일을 찾을 수 없습니다: {FILE_PATH}")