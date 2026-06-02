import pool from "../../../db/pool.js";
import redisClient from "../../../services/redis.service.js";

export async function getSensors() {

    console.log("전체 센서 조회 요청");
    
    const result = await pool.query(
        `
        SELECT
            s.sensor_id,
            s.lat,
            s.lon,
            s.grid_id,
            l.water_level,
            l.is_flooded,
            l.recorded_at
        FROM sensors s
        LEFT JOIN (
            SELECT DISTINCT ON (sensor_id)
                sensor_id,
                water_level,
                is_flooded,
                recorded_at
            FROM sensor_log
            ORDER BY sensor_id, recorded_at DESC
        ) l
        ON s.sensor_id = l.sensor_id
        `
    );

    console.log(`센서 ${result.rows.length}개 조회 완료`);

    return result.rows.map(row => ({
        id: row.sensor_id,
        lat: row.lat,
        lon: row.lon,
        gridId: row.grid_id,
        waterLevel: row.water_level,
        isFlooded: row.is_flooded,
        lastSeenAt: row.recorded_at
    }));
}
export async function ingest(data) {

    console.log("받은 데이터:", data);

    const {
        sensorId,
        waterLevel,
        isFlooded,
        measuredAt
    } = data;

    // sensor_id로 grid_id 조회
    const sensorResult = await pool.query(
        `
        SELECT grid_id
        FROM sensors
        WHERE sensor_id = $1
        `,
        [sensorId]
    );

    if (sensorResult.rows.length === 0) {
        console.error(`등록되지 않은 센서: ${sensorId}`);
        throw new Error("등록되지 않은 센서");
    }

    const gridId = sensorResult.rows[0].grid_id;
    console.log(`센서 ${sensorId} → grid ${gridId} 매핑 완료`);
    // sensor_log 저장
    await pool.query(
        `
        INSERT INTO sensor_log (
            sensor_id,
            grid_id,
            recorded_at,
            water_level,
            is_flooded
        )
        VALUES ($1, $2, $3, $4, $5)
        `,
        [
            sensorId,
            gridId,
            measuredAt,
            waterLevel,
            isFlooded
        ]
    );
    console.log("sensor_log 저장 완료");
    // Redis 갱신
    await redisClient.set(
        `sensor:${sensorId}`,
        JSON.stringify({
            id: sensorId,
            gridId,
            waterLevel,
            isFlooded,
            lastSeenAt: measuredAt
        })
    );
    console.log("Redis 캐시 갱신 완료");
    return {
        gridId,
        cached: true,
        wsNotified: false
    };
}