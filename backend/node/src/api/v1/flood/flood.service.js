import pool from '../../../db/pool.js';

const HORIZON_MAP = { current: 0, '1h': 1, '3h': 3, '6h': 6 };

export async function getHeatmapByHorizon(t) {
    const horizon = HORIZON_MAP[t] ?? 0;

    const { rows } = await pool.query(`
        SELECT 
            g.grid_id,
            g.center_lat AS lat,
            g.center_lon AS lon,
            ST_AsGeoJSON(g.geom)::json AS geom,
            p.is_flooded AS flood,
            p.target_time,
            p.model_version
        FROM flood_grid g
        JOIN flood_prediction p ON g.grid_id = p.grid_id
        WHERE p.predicted_at = (SELECT MAX(predicted_at) FROM flood_prediction)
          AND p.horizon = $1
          AND p.is_flooded = 1
    `, [horizon]);

    return {
        timestamp: new Date().toISOString(),
        horizon: t,
        type: 'FeatureCollection',
        features: rows.map(r => ({
            type: 'Feature',
            geometry: r.geom,
            properties: {
                grid_id: r.grid_id,
                lat: r.lat, 
                lon: r.lon,
                flood: r.flood,
                target_time: r.target_time,
                model_version: r.model_version
            }
        })),
        meta: { count: rows.length }
    };
}

export async function getGridDetails(gridId){
    const { rows } = await pool.query(`
        WITH latest_pred AS (
            SELECT predicted_at 
            FROM flood_prediction 
            WHERE grid_id = $1 
            ORDER BY predicted_at DESC 
            LIMIT 1
        )
        SELECT 
            g.grid_id,
            g.center_lat,
            g.center_lon,
            g.elevation,
            g.is_river,
            p.horizon,
            p.is_flooded,
            p.target_time,
            p.model_version
        FROM flood_grid g
        LEFT JOIN flood_prediction p ON g.grid_id = p.grid_id
        WHERE g.grid_id = $1
            AND p.predicted_at = (SELECT predicted_at FROM latest_pred)
        ORDER BY p.horizon ASC;
    `, [gridId]);
    
    if (rows.length === 0) {
        // 예측 데이터 없는 경우 - 격자 정보 확인 (추후 0로 돌려줘야할 수도 - 비가 0인 경우 등)
        const gridOnly = await pool.query('SELECT * FROM flood_grid WHERE grid_id = $1', [gridId]);
        return gridOnly.rows[0] ? { ...gridOnly.rows[0], predictions: [] } : null;
    }
    
    return {
        grid_id: rows[0].grid_id,
        info: {
            lat: rows[0].center_lat,
            lon: rows[0].center_lon,
            elevation: rows[0].elevation,
            is_river: rows[0].is_river === 1
        },

        predictions: rows.map(r => ({
            horizon: r.horizon,
            is_flooded: r.is_flooded === 1,
            target_time: r.target_time,
            model_version: r.model_version
        })),
        last_updated: rows[0].predicted_at
    };
}