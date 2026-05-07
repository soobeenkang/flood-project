CREATE TABLE IF NOT EXISTS flood_prediction (
    pred_id BIGSERIAL PRIMARY KEY,
    grid_id BIGINT NOT NULL REFERENCES flood_grid(grid_id),
    predicted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    target_time TIMESTAMPTZ NOT NULL,
    horizon SMALLINT NOT NULL,
    is_flooded INT NOT NULL,
    model_version VARCHAR(16)
);

CREATE INDEX IF NOT EXISTS idx_flood_pred_lookup
ON flood_prediction (predicted_at DESC, horizon);

CREATE INDEX IF NOT EXISTS idx_flood_pred_grid_time
    ON flood_prediction (grid_id, predicted_at DESC);

CREATE INDEX IF NOT EXISTS idx_flood_pred_flooded
    ON flood_prediction (predicted_at DESC, horizon)
    WHERE is_flooded = 1;

CREATE UNIQUE INDEX IF NOT EXISTS uq_flood_pred
    ON flood_prediction (grid_id, predicted_at, horizon);