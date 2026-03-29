"""
    서울시 aws 관측소의 위치 메타 데이터를 읽고
    기간과 위치 정보를 csv로 저장하는 코드
"""
import pandas as pd

meta = pd.read_csv("data/raw_aws/station_meta.csv.csv", encoding="cp949")
meta = meta.rename(columns={
    "지점": "station", 
    "시작일": "start_date",
    "종료일": "expired_date",
    "위도": "lat",
    "경도": "lon"
})

meta = meta[["station", "start_date", "expired_date", "lat", "lon"]]

print(f"관측소 수: {len(meta)}")
meta.to_csv("data/rainfall_history/station_coords.csv", index=False)

# 좌표 변경 이력이 있는 관측소 확인
changed = meta.groupby("station").filter(lambda x: len(x) > 1)
print(f"좌표 변경 관측소 수: {changed['station'].nunique()}")
print(changed[["station", "start_date", "expired_date", "lat", "lon"]])