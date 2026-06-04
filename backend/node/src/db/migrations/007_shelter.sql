CREATE TABLE IF NOT EXISTS shelter (
    shelter_id BIGSERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    address TEXT,
    type VARCHAR(50),
    geom GEOMETRY(Point, 4326) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_shelter_geom
    ON shelter USING GIST (geom);