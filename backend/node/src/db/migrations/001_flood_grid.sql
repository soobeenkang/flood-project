CREATE TABLE IF NOT EXISTS flood_grid (
    grid_id BIGINT PRIMARY KEY,
    geom GEOMETRY(Polygon, 4326) NOT NULL,
    center_lat FLOAT NOT NULL,
    center_lon FLOAT NOT NULL,
    elevation FLOAT,
    is_river SMALLINT DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_grid_geom 
ON flood_grid USING GIST (geom);

CREATE INDEX IF NOT EXISTS idx_grid_center
ON flood_grid (center_lat, center_lon);

COMMENT ON TABLE flood_grid IS '침수 예측 단위 격자 100m';