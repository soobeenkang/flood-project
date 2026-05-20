"""
flood_grid 테이블에 데이터 올리기

소스 데이터:
    - data/seoul_grid.geojson
    - data/seoul_grid_with_elevation.parquet
    - data/seoul_grid_with_river_flag.parquet
"""

import os
import geopandas as gpd
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "data"
GRID_FILE = f"{DATA_DIR}/grid/seoul_grid.geojson"
ELEV_FILE = f"{DATA_DIR}/seoul_grid_with_elevation.parquet"
RIVER_FILE = f"{DATA_DIR}/seoul_grid_with_river_flag.parquet"

DB_URL = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}"
    f"/{os.getenv('DB_NAME')}"
)

def main():
    print("GeoJSON 로드중...")
    gdf = gpd.read_file(GRID_FILE)
    print(f"   격자 {len(gdf):,}개")

    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        print(f"   CRS={gdf.crs} -> 4326 변환")
        gdf = gdf.to_crs(4326)

    print("   Parquet 로드중...")
    elev_df = pd.read_parquet(ELEV_FILE)
    river_df = pd.read_parquet(RIVER_FILE)
    print(f"   고도 {len(elev_df):,}행, 하천 {len(river_df):,}행")

    print(" 병합...")
    gdf = gdf.merge(elev_df, on="grid_id", how="left")
    gdf = gdf.merge(river_df, on="grid_id", how="left")

    gdf = gdf.rename(columns={
        "lat": "center_lat",
        "lon": "center_lon",
        "mean_elevation": "elevation",
    })

    gdf["is_river"] = gdf["is_river"].fillna(0).astype("int16")

    gdf = gdf.rename_geometry("geom")

    gdf = gdf[["grid_id", "geom", "center_lat", "center_lon", "elevation", "is_river"]]

    print(f"   준비 완료: {len(gdf):,}행")
    print(f"   결측치: elevation={gdf['elevation'].isna().sum()}, is_river={(gdf['is_river']==0).sum()}")

    engine = create_engine(DB_URL)

    with engine.begin() as conn:
        existing = conn.execute(text("SELECT COUNT(*) FROM flood_grid")).scalar()
        if existing > 0:
            print(f"  flood_grid에 이미 {existing:,}행 있음 -> TRUNCATE!")
            conn.execute(text("TRUNCATE TABLE flood_grid CASCADE;"))

        print("DB 업데이트 중...")
        gdf.to_postgis("flood_grid", engine, if_exists="append", index=False)

        with engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM flood_grid")).scalar()
            sample = conn.execute(text(
                "SELECT grid_id, ST_AsText(geom) as wkt, center_lat, center_lon, elevation, is_river "
                "FROM flood_grid LIMIT 1"
            )).fetchone()

        print(f"\n완료: flood_grid에 {count:,}행 업로드")
        print(f"  샘플: {sample}")


if __name__ == "__main__":
    main()
            