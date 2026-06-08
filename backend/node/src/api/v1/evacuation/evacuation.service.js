import pool from '../../../db/pool.js';
import redis from '../../../services/redis.service.js';

const FLOOD_PENALTY = 999999; // 침수 엣지 페널티 (m 단위 cost 기준)

// ── 침수 엣지 ID 수집 ─────────────────────────────────────────
//
// 우선순위: Redis 캐시(sensor) → sensor_log DB → flood_prediction
// 침수된 grid_id 를 모은 뒤, road_edge_grid 로 엣지 ID 로 변환.

async function getFloodedEdgeIds() {
  const floodedGridIds = new Set();

  // 1) Redis 캐시 스캔 (아두이노 센서 최우선)
  try {
    for await (const key of redis.scanIterator({
      match: 'sensor:grid:*',
      count: 200
    })) {
      const raw = await redis.get(key);
      if (!raw) continue;
      const data = JSON.parse(raw);
      if (data.isFlooded) floodedGridIds.add(key.split(':')[2]);
    }
  } catch (err) {
    console.warn('[route] Redis scan failed:', err.message);
  }

  // 2) Redis miss 된 그리드 → sensor_log DB fallback
  //    (Redis TTL 만료 등 대비, 최신 1건이 침수인 grid 추가)
  try {
    const { rows } = await pool.query(
      `SELECT DISTINCT ON (grid_id) grid_id, is_flooded
       FROM sensor_log
       ORDER BY grid_id, recorded_at DESC`,
    );
    rows.forEach((r) => {
      const id = String(r.grid_id);
      if (r.is_flooded && !floodedGridIds.has(id)) floodedGridIds.add(id);
    });
  } catch (err) {
    console.warn('[route] sensor_log fallback failed:', err.message);
  }

  // 3) flood_prediction (horizon=0) — 센서 없는 그리드 보완
  try {
    const { rows } = await pool.query(
      `SELECT DISTINCT ON (grid_id) grid_id
       FROM flood_prediction
       WHERE horizon = 0 AND is_flooded = 1
       ORDER BY grid_id, predicted_at DESC`,
    );
    rows.forEach((r) => floodedGridIds.add(String(r.grid_id)));
  } catch (err) {
    console.warn('[route] flood_prediction fallback failed:', err.message);
  }

  if (floodedGridIds.size === 0) return [];

  // grid_id → edge_id 변환
  const { rows: edgeRows } = await pool.query(
    `SELECT DISTINCT edge_id
     FROM road_edge_grid
     WHERE grid_id = ANY($1::bigint[])`,
    [[...floodedGridIds]],
  );

  return {
    edgeIds: edgeRows.map((r) => String(r.edge_id)),
    gridIdsSet: floodedGridIds
  };
}

// ── 좌표 → 가장 가까운 노드 ID ───────────────────────────────

async function nearestNode(lat, lon) {
  const { rows } = await pool.query(
    `SELECT from_node AS node_id
     FROM road_edge
     ORDER BY ST_StartPoint(geom) <-> ST_SetSRID(ST_MakePoint($2, $1), 4326)
     LIMIT 1`,
    [lat, lon],
  );
  return rows.length > 0 ? rows[0].node_id : null;
}

// ── pgr_aStar 실행 ────────────────────────────────────────────

async function runAStar(startNode, endNode, floodedEdgeIds) {
  const hasFlooded = floodedEdgeIds.length > 0;

  // 침수 엣지 ID 를 SQL 배열 리터럴로 바인딩
  // cost: 침수 엣지면 distance_m + FLOOD_PENALTY, 아니면 distance_m
  const sql = `
    SELECT
      path_seq,
      node,
      edge,
      agg_cost
    FROM pgr_aStar(
      $1::text,
      $2::bigint,
      $3::bigint,
      directed => false
    )
  `;

  // pgr_aStar 첫 번째 인자: 엣지 쿼리 문자열 (동적 생성)
  const safeEdgeIds = floodedEdgeIds.map((id) => Number(id)).join(',');
  const edgeQuery = hasFlooded
    ? `
      SELECT
        edge_id          AS id,
        from_node        AS source,
        to_node          AS target,
        distance_m + CASE
          WHEN edge_id = ANY(ARRAY[${safeEdgeIds}]::bigint[])
          THEN ${FLOOD_PENALTY}
          ELSE 0
        END              AS cost,
        distance_m + CASE
          WHEN edge_id = ANY(ARRAY[${safeEdgeIds}]::bigint[])
          THEN ${FLOOD_PENALTY}
          ELSE 0
        END              AS reverse_cost,
        ST_X(ST_StartPoint(geom)) AS x1,
        ST_Y(ST_StartPoint(geom)) AS y1,
        ST_X(ST_EndPoint(geom))   AS x2,
        ST_Y(ST_EndPoint(geom))   AS y2
      FROM road_edge
      `
    : `
      SELECT
        edge_id          AS id,
        from_node        AS source,
        to_node          AS target,
        distance_m       AS cost,
        distance_m       AS reverse_cost,
        ST_X(ST_StartPoint(geom)) AS x1,
        ST_Y(ST_StartPoint(geom)) AS y1,
        ST_X(ST_EndPoint(geom))   AS x2,
        ST_Y(ST_EndPoint(geom))   AS y2
      FROM road_edge
      `;

  const { rows } = await pool.query(sql, [edgeQuery, startNode, endNode]);
  return rows;
}

// ── 경로 엣지 geom → GeoJSON coordinates ─────────────────────

async function buildGeometry(validPathRows) {
  if (validPathRows.length === 0) return [];

  const edgeIds = validPathRows.map((r) => String(r.edge));

  const { rows } = await pool.query(
    `SELECT
       edge_id,
       from_node,
       to_node,
       ST_AsGeoJSON(geom)::json AS geom
     FROM road_edge
     WHERE edge_id = ANY($1::bigint[])`,
    [edgeIds],
  );

  // edge_id → coordinates 맵
  const geomMap = new Map(
    rows.map((r) => [String(r.edge_id), r]),
  );

  // 경로 순서대로 좌표 이어붙이기 (중복 첫점 제거)
  const coords = [];
  for (let i = 0; i < validPathRows.length; i++) {
    const currentRow = validPathRows[i];
    const edgeData = geomMap.get(String(currentRow.edge));
    if (!edgeData) continue;

    // 원본 좌표가 훼손되지 않도록 깊은 복사
    let segCoords = [...edgeData.geom.coordinates];

    if (currentRow.node === edgeData.to_node) {
      segCoords.reverse();
    }

    // 경로 레이어 구축 (접점 중복 제거하며 엮기)
    if (coords.length === 0) {
      coords.push(...segCoords);
    } else {
      coords.push(...segCoords.slice(1));
    }
  }
  return coords;
}

// ── 공개 서비스 함수 ──────────────────────────────────────────

export async function findEvacuationRoute(startLat, startLon, endLat, endLon, mode = 'avoid_flood') {
  // 병렬: 침수 엣지 수집 + 시작/끝 노드 탐색
  const [floodedData, startNode, endNode] = await Promise.all([
    getFloodedEdgeIds(),
    nearestNode(startLat, startLon),
    nearestNode(endLat, endLon),
  ]);

  if (!startNode || !endNode || startNode === endNode) return null;

  const floodedEdgeIds = floodedData?.edgeIds ?? [];
  const floodedGridIdsSet = floodedData?.gridIdsSet ?? new Set();

  const normalPathRows = await runAStar(startNode, endNode, []);
  const pathRows = await runAStar(startNode, endNode, floodedEdgeIds);
  if (pathRows.length === 0) return null;

  // 디버깅 로그
  /*
  console.log('====== [디버깅] 침수 라우팅 검증 ======');
  console.log('1. 수집된 침수 엣지 개수:', floodedEdgeIds.length);
  console.log('2. 수집된 침수 엣지 샘플:', floodedEdgeIds.slice(0, 5));
  */

  // --- [안전경로] -1 은 목적지 도착 행 (edge 없음) — 제외
  const validPathRows = [];
  pathRows.forEach((r) => {
    if (r.edge === -1 || r.edge == null) return;
    
    // 직전 엣지와 똑같은 엣지가 연속으로 들어오면 스킵
    if (validPathRows.length > 0 && validPathRows[validPathRows.length - 1].edge === r.edge) {
      return;
    }
    validPathRows.push(r);
  });

  const edgeIds = validPathRows.map((r) => String(r.edge));

  // -- [일반경로] 유효 엣지 정제 및 좌표 추출
  const validNormalPathRows = [];
  if (normalPathRows && normalPathRows.length > 0) {
    normalPathRows.forEach((r) => {
      if (r.edge === -1 || r.edge === '-1' || r.edge == null) return;
      if (validNormalPathRows.length > 0 && String(validNormalPathRows[validNormalPathRows.length - 1].edge) === String(r.edge))
        return;
      validNormalPathRows.push(r);
    });
  }

  let normalEdgeIds = validNormalPathRows.map((r) => String(r.edge));

  // 경로에 침수 구간 포함 여부
  const floodedEdgeSet = new Set(floodedEdgeIds);

  let bypassedCount = 0;

  // 모드에 따라 다르게 저장
  let targetEdgeIds = [];
  let targetRows = [];
  let hasFloodedSegment = false;
  let responseAvoidedGrids = 0;

  if (mode == 'fastest') {
    targetEdgeIds = normalEdgeIds;
    targetRows = validNormalPathRows;
    hasFloodedSegment = normalEdgeIds.some(id => floodedEdgeSet.has(id));
    responseAvoidedGrids = 0;
  } else {
    targetEdgeIds = edgeIds;
    targetRows = validPathRows;
    hasFloodedSegment = false;
    
    if (normalEdgeIds.length > 0) {
      const { rows: normalGridRows } = await pool.query(
        `SELECT DISTINCT grid_id FROM road_edge_grid WHERE edge_id = ANY($1::bigint[])`,
        [[...new Set(normalEdgeIds)]]
      );

      normalGridRows.forEach(r => {
        if (floodedGridIdsSet.has(String(r.grid_id))) {
          bypassedCount++;
        }
      });
    }

    responseAvoidedGrids = bypassedCount;
  }

  if (targetEdgeIds.length == 0) return null;
  const { rows: distRows } = await pool.query(
    `SELECT COALESCE(SUM(distance_m), 0) AS total_m FROM road_edge WHERE edge_id = ANY($1::bigint[])`,
    [targetEdgeIds],
  );
  const distanceM = parseFloat(distRows[0].total_m);

  const coordinates = await buildGeometry(targetRows);
  
  return {
    distanceM: Math.round(distanceM),
    hasFloodedSegment,
    avoidedGrids: responseAvoidedGrids,
    coordinates
  }
}

export default { findEvacuationRoute };