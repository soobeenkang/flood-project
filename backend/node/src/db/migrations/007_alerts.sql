CREATE TABLE alerts (
    id BIGSERIAL PRIMARY KEY,
    source_sn BIGINT UNIQUE, --조회할땐 제외되게
    region TEXT NOT NULL,
    type VARCHAR(20) NOT NULL,
    level VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    issued_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);