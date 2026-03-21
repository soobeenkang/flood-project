import zipfile
import glob
import os
import geopandas as gpd
import pandas as pd

 # 구간화 기준 설명:
    # - 0~100m   -> class 0
    # - 100~500m -> class 1
    # - 500m 초과 -> class 2

def generate_segokcheon_distance_class():
    # 기존 geojson grid 로드
    grid_path = "data/grid/gangnam_grid.geojson"

    print("기존 강남구 grid 로드 중...")
    grid = gpd.read_file(grid_path)

    if "grid_id" not in grid.columns:
        raise ValueError("'grid_id' 컬럼이 없습니다.")

    # 거리 계산은 미터 단위 좌표계에서 해야 하므로 EPSG:5179로 통일
    grid = grid.to_crs(epsg=5179)

    print("grid 개수:", len(grid))
    print("grid 범위:", grid.total_bounds)

    # DEM 데이터 로드 및 세곡천 필터링
    zip_folder = r"DEM data"
    extract_folder = r"DEM data"

    os.makedirs(extract_folder, exist_ok=True)

    print("zip 압축 해제 중...")
    zip_files = glob.glob(os.path.join(zip_folder, "*.zip"))

    for z in zip_files:
        with zipfile.ZipFile(z, "r") as zip_ref:
            zip_ref.extractall(extract_folder)

    print("압축 해제 완료")

    shp_files = glob.glob(os.path.join(extract_folder, "**/*.shp"), recursive=True)

    gdfs = []
    print("shp 파일 로드 중...")
    for f in shp_files:
        try:
            temp = gpd.read_file(f)
            gdfs.append(temp)
        except Exception as e:
            print("읽기 실패:", f, e)

    if not gdfs:
        raise ValueError("읽어온 shp 파일이 없습니다.")

    gdf_all = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    print("전체 데이터 수:", len(gdf_all))
    print("전체 컬럼:", list(gdf_all.columns))

    if "하천명" not in gdf_all.columns:
        raise ValueError("'하천명' 컬럼이 없습니다.")

    
    segokcheon = gdf_all[
        gdf_all["하천명"].astype(str).str.contains("세곡천", na=False)
        & gdf_all.geometry.notnull()
    ].copy()

    print("세곡천 원본 개수:", len(segokcheon))

    if len(segokcheon) == 0:
        raise ValueError("세곡천 데이터가 없습니다. 하천명 값을 확인하세요.")

    print("세곡천 geometry 타입(원본):")
    print(segokcheon.geometry.type.value_counts())


    # 좌표계 통일 - 원본 shp에 crs가 없을 때만 강제 지정
    if segokcheon.crs is None:
        segokcheon = segokcheon.set_crs(epsg=5186)

    segokcheon = segokcheon.to_crs(epsg=5179)

    print("세곡천 범위:", segokcheon.total_bounds)

   
    # Polygon -> boundary 변환
    # 세곡천이 Polygon으로 들어와 있으므로,
    # polygon 자체까지의 거리 대신 polygon 경계선(boundary)까지의 거리를 계산한다.
    segokcheon_boundary = segokcheon.copy()
    segokcheon_boundary["geometry"] = segokcheon_boundary.geometry.boundary
    segokcheon_boundary = segokcheon_boundary[
        ~segokcheon_boundary.geometry.is_empty
    ].copy()

    if len(segokcheon_boundary) == 0:
        raise ValueError("세곡천 boundary 변환 후 geometry가 비었습니다.")

    print("세곡천 geometry 타입(boundary 변환 후):")
    print(segokcheon_boundary.geometry.type.value_counts())

  
    # grid centroid 기준 최근접 거리 계산
   
    print("grid centroid 계산 중...")
    centroids = grid.geometry.centroid

    centroid_gdf = gpd.GeoDataFrame(
        {"grid_id": grid["grid_id"]},
        geometry=centroids,
        crs="EPSG:5179"
    )

    print("최근접 세곡천 경계선 거리 계산 중...")
    nearest = gpd.sjoin_nearest(
        centroid_gdf,
        segokcheon_boundary[["geometry"]],
        how="left",
        distance_col="dist_to_segokcheon_boundary"
    )

    print("최근접 세곡천 경계선 매핑 수:", len(nearest))
    print(nearest["dist_to_segokcheon_boundary"].describe())

   
    # 거리 구간화
    result = nearest[["grid_id", "dist_to_segokcheon_boundary"]].copy()

    max_dist = result["dist_to_segokcheon_boundary"].max()
    if pd.isna(max_dist):
        raise ValueError("dist_to_segokcheon_boundary 전체가 NaN입니다. 좌표계나 geometry를 확인하세요.")

    result["dist_to_segokcheon_boundary"] = result["dist_to_segokcheon_boundary"].fillna(max_dist)

    # 구간화 기준 설명:
    # - 0~100m   -> class 0
    # - 100~500m -> class 1
    # - 500m 초과 -> class 2
   
    bins = [-1, 100, 500, float("inf")]

    result["dist_to_segokcheon_class"] = pd.cut(
        result["dist_to_segokcheon_boundary"],
        bins=bins,
        labels=[0, 1, 2]
    ).astype(int)

    # 최종 저장 컬럼은 grid_id와 dist_to_segokcheon_class만 남김
    result = result[["grid_id", "dist_to_segokcheon_class"]].copy()

    print("구간별 개수:")
    print(result["dist_to_segokcheon_class"].value_counts().sort_index())

    # parquet 저장
    os.makedirs("data/grid", exist_ok=True)
    output_path = "data/grid/segokcheon_distance_class.parquet"
    result.to_parquet(output_path, index=False)

    print("완료:", output_path)
    return result


if __name__ == "__main__":
    generate_segokcheon_distance_class()