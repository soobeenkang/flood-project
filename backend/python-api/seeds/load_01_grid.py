import os
import geopandas as gpd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv(encoding="utf-8")

DB_URL = (
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}"
    f"/{os.getenv('DB_NAME')}"
)

engine = create_engine(DB_URL)

gdf = gpd.read_file("data/seoul_grid.geojson")

if gdf.crs is None:
    gdf = gdf.set_crs(epsg=4326)
elif gdf.crs.to_epsg() != 4326:
    gdf = gdf.to_crs(epsg=4326)

required_cols = ["grid_id", "lon", "lat", "geometry"]
gdf = gdf.dropna(subset=required_cols)

records = []
for _, row in gdf.iterrows():
    records.append({
        "grid_id": int(row["grid_id"]),
        "center_lng": float(row["lon"]),
        "center_lat": float(row["lat"]),
        "geom_wkt": row["geometry"].wkt,
        "elevation": None,
        "is_river": False
    })

print(f"적재할 grid 개수: {len(records)}")

with engine.begin() as conn:
    conn.execute(text("""
        INSERT INTO flood_grid (
            grid_id,
            geom,
            center_lat,
            center_lng,
            elevation,
            is_river
        )
        VALUES (
            :grid_id,
            ST_GeomFromText(:geom_wkt, 4326),
            :center_lat,
            :center_lng,
            :elevation,
            :is_river
        )
        ON CONFLICT (grid_id) DO NOTHING
    """), records)

print("flood_grid 적재 완료")