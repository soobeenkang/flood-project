import pool from '../../../db/pool.js';
import redis from '../../../services/redis.service.js';

const HORIZON_MAP = { now: 0, '1h': 1, '3h': 3, '6h': 6 };
const HORIZON_LABEL = { 0: 'now', 1: '1h', 3: '3h', 6: '6h' };
const HORIZONS      = ['now', '1h', '3h', '6h'];
const SENSOR_CACHE_TTL = 300;

// ── 센서 데이터 조회 (Redis -> sensor_log DB) ─────────

/**
 * Redis 캐시 miss시 sensor_log DB에서 최신 1건 fallback.
 * DB에서 읽으면 Redis에 재캐싱.
 *
 * @param {bigint|string} gridId
 * @returns {Promise<{isFlooded:boolean, waterLevel:number, measuredAt:string}|null>}
 */
async function getSensorData(gridId) {
  try {
    const raw = await redis.get(`sensor:grid:${gridId}`);
    if (raw) return JSON.parse(raw);
  } catch (err) {
    console.warn(`[heatmap] Redis get failed for grid ${gridId}:`, err.message);
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
    const data = {
      isFlooded:  is_flooded,
      waterLevel: water_level,
      measuredAt: recorded_at.toISOString(),
    };

    try {
      await redis.set(`sensor:grid:${gridId}`, JSON.stringify(data), { EX: SENSOR_CACHE_TTL });
    } catch (_) { /* 재캐싱 실패 무시 */ }

    return data;
  } catch (err) {
    console.error(`[heatmap] sensor_log fallback failed for grid ${gridId}:`, err.message);
    return null;
  }
}

// ── 그리드 행 -> 응답 객체 변환 헬퍼 ─────────────────────────
function buildGridItem(row, sensor) {
  if (sensor) {
    return {
      id:         String(row.grid_id),
      lat:        row.center_lat,
      lon:        row.center_lon,
      isFlooded:  sensor.isFlooded,
      waterLevel: sensor.waterLevel,
      measuredAt: sensor.measuredAt,
      source:     'sensor',
    };
  }
  if (row.predicted_at != null) {
    return {
      id:          String(row.grid_id),
      lat:         row.center_lat,
      lon:         row.center_lon,
      isFlooded:   row.is_flooded === 1,
      predictedAt: row.predicted_at ?? null,
      source:      'prediction',
    };
  }
  // 아무런 값 없는 그리드
  return {
    id:          String(row.grid_id),
    lat:         row.center_lat,
    lon:         row.center_lon,
    isFlooded:   false,
    source:      'none',
  };
}

// ── 공통 쿼리: 반경 내 그리드 + 특정 horizon 예측값 ──────────
async function fetchGridRows(lat, lon, radius, horizonNum) {
  const { rows } = await pool.query(
    `SELECT
       fg.grid_id,
       fg.center_lat,
       fg.center_lon,
       fp.is_flooded,
       fp.predicted_at
     FROM flood_grid fg
     LEFT JOIN LATERAL (
       SELECT is_flooded, predicted_at
       FROM flood_prediction
       WHERE grid_id = fg.grid_id AND horizon = $3
       ORDER BY predicted_at DESC
       LIMIT 1
     ) fp ON TRUE
     WHERE ST_DWithin(
       fg.geom::geography,
       ST_SetSRID(ST_MakePoint($2, $1), 4326)::geography,
       $4
     )
     ORDER BY ST_Distance(
       fg.geom::geography,
       ST_SetSRID(ST_MakePoint($2, $1), 4326)::geography
     )`,
    [lat, lon, horizonNum, radius],
  );
  return rows;
}

// ── 서비스 함수 ───────────────────────────────────────────────

/**
 * 특정 horizon 히트맵 그리드 목록.
 * 센서(Redis -> sensor_log) 우선, 없으면 flood_prediction.
 */
export async function getGrids(lat, lon, radius, horizon) {
  const rows = await fetchGridRows(lat, lon, radius, HORIZON_MAP[horizon] ?? 0);

  const grids = await Promise.all(
    rows.map(async (row) => {
      const sensor = await getSensorData(row.grid_id);
      return buildGridItem(row, sensor);
    }),
  );

  return { horizon, grids };
}

/**
 * 모든 horizon(now·1h·3h·6h) 데이터를 한 번에 반환.
 * 프론트에서 시간대 버튼 전환 시 재요청 없이 클라이언트 캐시로 처리 가능.
 *
 * 센서 데이터는 horizon 무관하게 동일하므로 그리드당 1회만 조회.
 */
export async function getAllHorizons(lat, lon, radius) {
  // 4개 horizon 쿼리 병렬 실행
  const [nowRows, h1Rows, h3Rows, h6Rows] = await Promise.all(
    [0, 1, 3, 6].map((h) => fetchGridRows(lat, lon, radius, h)),
  );

  const rowsByHorizon = { now: nowRows, '1h': h1Rows, '3h': h3Rows, '6h': h6Rows };

  // 전체 grid_id 중복 제거 -> 센서 조회 1회
  const allGridIds = [...new Set(nowRows.map((r) => String(r.grid_id)))];
  const sensorMap  = new Map(
    await Promise.all(
      allGridIds.map(async (id) => [id, await getSensorData(id)]),
    ),
  );

  const result = {};
  for (const horizon of HORIZONS) {
    result[horizon] = rowsByHorizon[horizon].map((row) => {
      const sensor = sensorMap.get(String(row.grid_id));
      return buildGridItem(row, sensor);
    });
  }

  return result;
}

/**
 * 특정 그리드 상세 (모든 horizon + 센서 데이터).
 */
export async function getGrid(gridId) {
  const gid = BigInt(gridId);

  const { rows: gridRows } = await pool.query(
    `SELECT grid_id, center_lat, center_lon, elevation
     FROM flood_grid WHERE grid_id = $1`,
    [gid],
  );
  if (gridRows.length === 0) return null;

  const grid = gridRows[0];

  const { rows: predRows } = await pool.query(
    `SELECT DISTINCT ON (horizon) horizon, is_flooded, predicted_at
     FROM flood_prediction
     WHERE grid_id = $1
     ORDER BY horizon, predicted_at DESC`,
    [gid],
  );

  const horizons = {};
  predRows.forEach((r) => {
    const label = HORIZON_LABEL[r.horizon] ?? String(r.horizon);
    horizons[label] = { isFlooded: r.is_flooded === 1, predictedAt: r.predicted_at };
  });

  const sensor    = await getSensorData(gid);
  const isFlooded = sensor ? sensor.isFlooded : (horizons.now?.isFlooded ?? false);

  return {
    id:          String(grid.grid_id),
    lat:         grid.center_lat,
    lon:         grid.center_lon,
    elevation:   grid.elevation,
    isFlooded,
    source:      sensor ? 'sensor' : 'prediction',
    predictedAt: horizons.now?.predictedAt ?? null,
    horizons,
    ...(sensor && { sensorData: sensor }),
  };
}

export default { getGrids, getAllHorizons, getGrid };