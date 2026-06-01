CREATE TABLE IF NOT EXISTS shelter (
    shelter_id BIGSERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    address TEXT,
    geom GEOMETRY(Point, 4326) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_shelter_geom
    ON shelter USING GIST (geom);

CREATE INDEX IF NOT EXISTS idx_shelter_active
    ON shelter (is_active) WHERE is_active = TRUE;