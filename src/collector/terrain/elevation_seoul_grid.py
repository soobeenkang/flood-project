import os
import geopandas as gpd
import pandas as pd


def map_elevation_to_seoul_grid():
    # -----------------------------
    # 1) 파일 경로 설정
    # -----------------------------
    contour_shp_path = r"DEM data/N3L_F0010000_11.shp"
    grid_path = r"data/grid/seoul_grid.geojson"
    output_path = r"data/grid/seoul_grid_with_elevation.parquet"

    # -----------------------------
    # 2) 서울시 grid 로드
    # -----------------------------
    print("서울시 grid 로드 중...")
    grid = gpd.read_file(grid_path)

    if "grid_id" not in grid.columns:
        raise ValueError("'grid_id' 컬럼이 없습니다.")

    if grid.crs is None:
        raise ValueError("seoul_grid.geojson의 CRS가 없습니다.")

    # 거리/면적/공간연산은 미터 단위 좌표계에서 하는 게 안전하므로 5179로 통일
    grid = grid.to_crs(epsg=5179)

    print("grid 개수:", len(grid))
    print("grid CRS:", grid.crs)
    print("grid bounds:", grid.total_bounds)

    # -----------------------------
    # 3) 등고선 shp 로드
    # -----------------------------
    print("\n등고선 shp 로드 중...")
    contours = gpd.read_file(contour_shp_path)

    if contours.crs is None:
        raise ValueError("등고선 shp의 CRS가 없습니다. .prj 파일을 확인하세요.")

    print("등고선 데이터 개수:", len(contours))
    print("등고선 CRS:", contours.crs)
    print("등고선 컬럼:", list(contours.columns))

    # 연속수치지형도 설명서 기준 N3L_F0010000는 1:5,000 등고선 레이어다.
    # 실제 고도값은 CONT 컬럼에 들어있는 것으로 보고 사용한다.
    # 레이어명 규칙과 등고선 레이어 표는 설명서에 나와 있다. :contentReference[oaicite:1]{index=1}
    if "CONT" not in contours.columns:
        raise ValueError("'CONT' 컬럼이 없습니다. 등고선 레이어가 맞는지 확인하세요.")

    # -----------------------------
    # 4) 필요한 데이터만 필터링
    # -----------------------------
    contours = contours[
        contours.geometry.notnull()
        & contours.geometry.type.isin(["LineString", "MultiLineString"])
    ].copy()

    if len(contours) == 0:
        raise ValueError("사용 가능한 등고선 geometry가 없습니다.")

    # CONT를 숫자형 고도값으로 변환
    contours["elevation"] = pd.to_numeric(contours["CONT"], errors="coerce")
    contours = contours[contours["elevation"].notnull()].copy()

    if len(contours) == 0:
        raise ValueError("CONT 컬럼에서 숫자형 고도값을 추출하지 못했습니다.")

    print("\n필터링 후 등고선 개수:", len(contours))
    print("elevation 예시:", contours["elevation"].head().tolist())

    # -----------------------------
    # 5) 좌표계 통일
    # -----------------------------
    contours = contours.to_crs(epsg=5179)

    print("좌표계 통일 후 contours CRS:", contours.crs)
    print("contours bounds:", contours.total_bounds)

    # -----------------------------
    # 6) 등고선 vertex를 포인트로 변환
    # -----------------------------
    # 각 등고선의 좌표점(vertex)에 해당 등고값을 부여해서
    # grid 안에 들어간 점들의 평균 고도를 계산한다.
    print("\n등고선 vertex를 포인트로 변환 중...")

    points = []

    for _, row in contours.iterrows():
        geom = row.geometry
        elev = row["elevation"]

        if geom.geom_type == "LineString":
            lines = [geom]
        elif geom.geom_type == "MultiLineString":
            lines = geom.geoms
        else:
            continue

        for line in lines:
            for x, y in line.coords:
                points.append((x, y, elev))

    if not points:
        raise ValueError("등고선에서 추출된 포인트가 없습니다.")

    df_points = pd.DataFrame(points, columns=["x", "y", "elevation"])

    points_gdf = gpd.GeoDataFrame(
        df_points,
        geometry=gpd.points_from_xy(df_points["x"], df_points["y"]),
        crs="EPSG:5179"
    )

    print("추출된 포인트 개수:", len(points_gdf))
    print("포인트 bounds:", points_gdf.total_bounds)

    # -----------------------------
    # 7) grid와 spatial join
    # -----------------------------
    print("\n공간 매핑 중...")
    joined = gpd.sjoin(
        points_gdf[["elevation", "geometry"]],
        grid[["grid_id", "geometry"]],
        how="inner",
        predicate="intersects"
    )

    print("매핑된 포인트 수:", len(joined))

    if len(joined) == 0:
        raise ValueError("grid와 매핑된 포인트가 없습니다. 좌표계나 데이터 범위를 확인하세요.")

    # -----------------------------
    # 8) grid별 평균 고도 계산
    # -----------------------------
    print("\ngrid별 평균 고도 계산 중...")
    elevation_per_grid = (
        joined.groupby("grid_id")["elevation"]
        .mean()
        .reset_index()
        .rename(columns={"elevation": "mean_elevation"})
    )

    print("고도 계산된 grid 수:", len(elevation_per_grid))

    # -----------------------------
    # 9) grid에 merge
    # -----------------------------
    result = grid.merge(elevation_per_grid, on="grid_id", how="left")

    # 결측값 처리
    # 등고선 포인트가 없는 grid는 우선 전체 평균으로 채운다.
    # 나중에 더 정교하게 하고 싶으면 nearest interpolation으로 바꿀 수 있다.
    overall_mean = result["mean_elevation"].mean()

    if pd.isna(overall_mean):
        raise ValueError("전체 평균 고도 계산이 실패했습니다.")

    result["mean_elevation"] = result["mean_elevation"].fillna(overall_mean)

    print("\n최종 고도 통계:")
    print(result["mean_elevation"].describe())

    # -----------------------------
    # 10) parquet 저장
    # -----------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 필요한 컬럼만 저장
    result[["grid_id", "mean_elevation"]].to_parquet(output_path, index=False)

    print("\n완료:", output_path)
    return result[["grid_id", "mean_elevation"]]


if __name__ == "__main__":
    map_elevation_to_seoul_grid()