import zipfile
import glob
import os
import geopandas as gpd
import pandas as pd

# 경로 설정
zip_folder = r"DEM data"
extract_folder = r"DEM data"

os.makedirs(extract_folder, exist_ok=True)

# zip 압축 해제
zip_files = glob.glob(os.path.join(zip_folder, "*.zip"))

for z in zip_files:
    with zipfile.ZipFile(z, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

print("압축 해제 완료")

# 모든 shp 파일 찾기
shp_files = glob.glob(os.path.join(extract_folder, "**/*.shp"), recursive=True)

# 모든 shp 하나로 합치기
gdfs = []
for f in shp_files:
    try:
        gdf = gpd.read_file(f)
        gdfs.append(gdf)
    except:
        print("읽기 실패:", f)

gdf_all = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
print("전체 데이터 수:", len(gdf_all))


# 등고선만 필터링
contours = gdf_all[
    (gdf_all.geometry.type.isin(["LineString", "MultiLineString"])) &
    (gdf_all["등고수치"].notnull())
].copy()

print("등고선 개수:", len(contours))


# (x, y, elev) 추출
points = []

for _, row in contours.iterrows():
    geom = row.geometry
    elev = row["등고수치"]

    if geom.geom_type == "LineString":
        lines = [geom]
    elif geom.geom_type == "MultiLineString":
        lines = geom.geoms
    else:
        continue

    for line in lines:
        for x, y in line.coords:
            points.append((x, y, elev))

print("포인트 개수:", len(points))


# DataFrame으로 변환
df_points = pd.DataFrame(points, columns=["x", "y", "elevation"])

print(df_points.head())


# csv 파일로 저장
df_points.to_csv("contour_points.csv", index=False)

print("완료: contour_points.csv 저장됨")