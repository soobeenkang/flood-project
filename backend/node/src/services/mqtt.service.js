// client.subscribe("flood/data");

// const mqtt = require("mqtt");
// const { Pool } = require("pg");
import dotenv from "dotenv";
import mqtt from "mqtt";
import pkg from "pg";
dotenv.config();

const { Pool } = pkg;

const pool = new Pool({
    host: process.env.DB_HOST,
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    database: process.env.DB_NAME,
    port: process.env.DB_PORT
});

export function initMqtt() {

    const client = mqtt.connect("mqtt://mosquitto:1883");

    client.on("connect", () => {

        console.log("MQTT 연결 성공");

        client.subscribe("flood/data");
    });

    client.on("message", async (topic, message) => {

        try {

            const data = JSON.parse(message.toString());

            console.log(data);
//
            // MQTT 데이터
            const sensorId = data.sensor_id;
            const waterLevel = data.water;
            const lat = data.lat;
            const lng = data.lng;

            // 침수 여부 예시
            const isFlooded = waterLevel > 1000;

            // 위경도로 grid_id 찾기
            const gridResult = await pool.query(
                `
                SELECT grid_id
                FROM flood_grid
                WHERE ST_Contains(
                    geom,
                    ST_SetSRID(ST_Point($1, $2), 4326)
                )
                LIMIT 1
                `,
                [lng, lat]
            );

            if (gridResult.rows.length === 0) {

                console.log("해당 위치의 grid 없음");
                return;
            }

            const gridId = gridResult.rows[0].grid_id;

            // sensor_log 저장
            await pool.query(
                `
                INSERT INTO sensor_log
                (sensor_id, grid_id, water_level, is_flooded)
                VALUES ($1, $2, $3, $4)
                `,
                [sensorId, gridId, waterLevel, isFlooded]
            );

            console.log("DB 저장 완료");

        } catch (err) {

            console.error(err);
        }
    });
}