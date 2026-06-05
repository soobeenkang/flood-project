CREATE TABLE IF NOT EXISTS subscriptions (
    subscription_id BIGSERIAL PRIMARY KEY,
    grid_id BIGINT NOT NULL REFERENCES flood_grid(grid_id),
    email_encrypted TEXT NOT NULL,
    email_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sub_grid
    ON subscriptions (grid_id);

CREATE UNIZUE INDEX IF NOT EXISTS uq_sub_grid_email
    ON subscriptions (grid_id, email_hash);

COMMENT ON TABLE subscriptions IS '그리드 침수 이메일 알림 구독 - 이메일 AES-256 암호화';