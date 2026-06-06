CREATE TABLE IF NOT EXISTS seoul_weather (
    grid_id INTEGER NOT NULL,    -- 서울 격자 
    tmfc TIMESTAMP NOT NULL,    -- 발표시각(YYYY-MM-DD HH:MM:SS)

    -- 실황(현재)
    rn1_now FLOAT,    -- 현재 강수량(mm)
    t1h_now FLOAT,    -- 현재 기온(℃)
    vec_now FLOAT,    -- 현재 풍향(°)
    wsd_now FLOAT,    -- 현재 풍속(m/s)

    -- 초단기예보
    rn1_1h FLOAT,    -- 1시간 후의 예측 강수량(mm)
    rn1_2h FLOAT,    -- 2시간 후의 예측 강수량(mm)
    rn1_3h FLOAT,    -- 3시간 후의 예측 강수량(mm)
    rn1_4h FLOAT,    -- 4시간 후의 예측 강수량(mm)
    rn1_5h FLOAT,    -- 5시간 후의 예측 강수량(mm)
    rn1_6h FLOAT,    -- 6시간 후의 예측 강수량(mm)

    sky_1h FLOAT,    -- 1시간 후 하늘 상태(1=맑음, 2=구름조금, 3=구름많음, 4=흐림)

    PRIMARY KEY (grid_id, tmfc)
);
CREATE INDEX IF NOT EXISTS idx_seoul_weather_tmfc
ON seoul_weather (tmfc DESC);

CREATE INDEX IF NOT EXISTS idx_weather_grid_tmfc
ON seoul_weather (grid_id, tmfc DESC);

COMMENT ON TABLE seoul_weather IS '서울시 100m 격자별 기상청 실황 및 초단기 예보 데이터';
