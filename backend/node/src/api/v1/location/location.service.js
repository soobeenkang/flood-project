const pool = require('../../../db/pool');
const { client: redis } = require('../../../db/redis');

/**
 * Redis 센서 캐시 조회.
 * key: sensor:grid:{gridId}
 * @param {bigint|string} gridId
 * @returns {Promise<{isFlooded:boolean, waterLevel:number, measuredAt:string}|null>}
 */
async function getSensorData(gridId) {
  try {
    const raw = await redis.get(`sensor:grid:${gridId}`);
    if (raw)
      return JSON.parse(raw);
  } catch (err) {
    console.warn('[location] Redis get failed for grid ${gridId}:', err.message);
  }

  try {
    const { rows } = await pool.query(
      `SELECT water_level, is_flooded, recorded_at
      FROM sensor_log
      WHERE grid_id = $1
      ORDER BY recorded_at DESC
      LIMIT 1`,
      [BigInt(gridId)],
    );
    if (rows.length === 0) return null;

    const { water_level, is_flooded, recorded_at } = rows[0];
    return {
      isFlooded: is_flooded,
      waterLevel: water_level,
      measuredAt: recorded_at.toISOString(),
    };
  } catch (err) {
    console.error('[location] sensor_log query failed for grid $[gridId]:', err.message);
    return null;
  }
}

/**
 * 좌표가 속한 그리드의 침수 여부 반환.
 * 센서 캐시 -> sensor_log -> flood_prediction 순으로 우선 적용.
 *
 * @param {number} lat
 * @param {number} lon
 * @returns {Promise<{isFlooded:boolean, gridId:string|null, source:string}>}
 */
export async function checkFloodAtPoint(lat, lon) {
  const { rows } = await pool.query(
    `SELECT grid_id
     FROM flood_grid
     WHERE ST_Contains(geom, ST_SetSRID(ST_MakePoint($2, $1), 4326))
     LIMIT 1`,
    [lat, lon],
  );

  if (rows.length === 0) {
    return { isFlooded: false, gridId: null, source: 'none' };
  }

  const gridId = rows[0].grid_id;

  const sensor = await getSensorData(gridId);
  if (sensor) {
    return { isFlooded: sensor.isFlooded, gridId: String(gridId), source: 'sensor' };
  }

  const { rows: predRows } = await pool.query(
    `SELECT is_flooded
     FROM flood_prediction
     WHERE grid_id = $1 AND horizon = 0
     ORDER BY predicted_at DESC
     LIMIT 1`,
    [gridId],
  );

  const isFlooded = predRows.length > 0 && predRows[0].is_flooded === 1;
  return { isFlooded, gridId: String(gridId), source: 'prediction' };
}

module.exports = { checkFloodAtPoint };