import pool from "../../../db/pool.js";
import redisClient from "../../../services/redis.service.js";

export async function getSensor(sensorId){
    const cache =
        await redisClient.get(`sensor:${sensorId}`);

    if (cache) {
        console.log("Redis 조회");
        return JSON.parse(cache);
    }

    console.log("DB 조회");

    const result = await pool.query(
        `
        SELECT *
        FROM sensor_log
        WHERE sensor_id = $1
        ORDER BY recorded_at DESC
        LIMIT 1
        `,
        [sensorId]
    );

    if(result.rows.length==0){
        return null;
    }

    const sensor = result.rows[0];

    //redis에 저장하는거 
    await redisClient.set(
        `sensor:${sensorId}`,
        JSON.stringify(sensor)
    );

    return sensor;
}