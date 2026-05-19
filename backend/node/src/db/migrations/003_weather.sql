CREATE TABLE IF NOT EXISTS weather_observation (
    obs_id BIGSERIAL PRIMARY KEY,
    time TIMESTAMPTZ NOT NULL,
    station_id VARCHAR(16) NOT NULL,
    rainfall_1h FLOAT,
    rainfall_3h FLOAT,
    rainfall_6h FLOAT
);
CREATE INDEX IF NOT EXISTS idx_weather_time
ON weather_observation (time DESC);

CREATE INDEX IF NOT EXISTS idx_weather_station_time
ON weather_observation (station_id, time DESC);

CREATE UNIQUE INDEX IF NOT EXISTS uq_weather_station_time
ON weather_observation (station_id, time);

COMMENT ON TABLE weather_observation IS '관측소별 강수 예보 데이터'