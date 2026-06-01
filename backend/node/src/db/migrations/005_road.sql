CREATE EXTENSION IF NOT EXISTS postgis;

-- 침수 grid
CREATE TABLE IF NOT EXISTS flood_grid (
    grid_id BIGSERIAL PRIMARY KEY,

    geom GEOMETRY(Polygon, 4326),

    center_lat FLOAT,
    center_lng FLOAT,

    elevation FLOAT,

    is_river BOOLEAN DEFAULT FALSE,

    risk_score FLOAT DEFAULT 0
);

-- 도로 링크 (A* 그래프)
CREATE TABLE IF NOT EXISTS road_edge (
    edge_id BIGSERIAL PRIMARY KEY,

    from_node BIGINT NOT NULL,
    to_node BIGINT NOT NULL,

    geom GEOMETRY(LineString, 4326) NOT NULL,

    distance_m FLOAT NOT NULL,

    flood_weight FLOAT DEFAULT 0,

    road_type VARCHAR(20)
);

-- 도로 링크 ↔ 침수 grid 매핑
CREATE TABLE IF NOT EXISTS road_edge_grid (
    edge_id BIGINT REFERENCES road_edge(edge_id),
    grid_id BIGINT REFERENCES flood_grid(grid_id),

    PRIMARY KEY (edge_id, grid_id)
);

CREATE INDEX IF NOT EXISTS idx_flood_grid_geom
ON flood_grid USING GIST (geom);

CREATE INDEX IF NOT EXISTS idx_road_edge_geom
ON road_edge USING GIST (geom);

CREATE INDEX IF NOT EXISTS idx_road_edge_from
ON road_edge(from_node);

CREATE INDEX IF NOT EXISTS idx_road_edge_to
ON road_edge(to_node);