CREATE TABLE IF NOT EXISTS sensors (
    sensor_id VARCHAR(20) PRIMARY KEY,
    lat DOUBLE PRECISION NOT NULL,
    lon DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS sensor_log (
    log_id BIGSERIAL PRIMARY KEY,
    sensor_id VARCHAR(20) NOT NULL REFERENCES sensors(sensor_id),
    grid_id BIGINT NOT NULL REFERENCES flood_grid(grid_id),
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    water_level FLOAT NOT NULL,
    is_flooded BOOLEAN DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_log_sensor_time
    ON sensor_log (sensor_id, recorded_at DESC);

CREATE INDEX IF NOT EXISTS idx_log_grid_time
    ON sensor_log (grid_id, recorded_at DESC);

COMMENT ON TABLE sensor_log IS '아두이노 수위 센서 측정 이력 - 그리드 당';