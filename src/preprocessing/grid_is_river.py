import geopandas as gpd
import os


def mark_river_grid():
    # -----------------------------
    # 1) 파일 경로
    # -----------------------------
    grid_path = r"data/grid/seoul_grid.geojson"
    river_path = r"C:\Users\dlwod\flood-project\DEM data\river\N3A_E0032111.shp"
    output_path = r"data/grid/seoul_grid_with_river_flag.parquet"

    # -----------------------------
    # 2) grid 로드
    # -----------------------------
    print("grid 로드 중...")
    grid = gpd.read_file(grid_path)

    if "grid_id" not in grid.columns:
        raise ValueError("grid_id 컬럼 없음")

    if grid.crs is None:
        raise ValueError("grid CRS 없음")

    # 거리/공간 연산용 좌표계
    grid = grid.to_crs(epsg=5179)

    print("grid 개수:", len(grid))
    print("grid bounds:", grid.total_bounds)

    # -----------------------------
    # 3) 실폭하천 로드
    # -----------------------------
    print("\n실폭하천 로드 중...")
    river = gpd.read_file(river_path)

    if river.crs is None:
        raise ValueError("하천 shp CRS 없음")

    river = river.to_crs(epsg=5179)

    print("하천 개수:", len(river))
    print("하천 bounds:", river.total_bounds)
    print("geometry 타입:", river.geometry.type.unique())

    # -----------------------------
    # 4) geometry 병합 (속도 핵심)
    # -----------------------------
    print("\n하천 geometry 병합 중...")
    river_union = river.geometry.union_all()

    # -----------------------------
    # 5) grid와 겹치는지 판별
    # -----------------------------
    print("is_river 계산 중...")
    grid["is_river"] = grid.intersects(river_union).astype(int)

    print("\n=== 결과 ===")
    print(grid["is_river"].value_counts())

    # -----------------------------
    # 6) 저장
    # -----------------------------
    os.makedirs("data/grid", exist_ok=True)

    result = grid[["grid_id", "is_river"]].copy()
    result.to_parquet(output_path, index=False)

    print("\n완료:", output_path)

    return result


if __name__ == "__main__":
    mark_river_grid()