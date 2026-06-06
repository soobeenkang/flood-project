// Base URL 환경변수로 관리
// .env 파일에 VITE_API_BASE_URL=http://localhost:3000 추가 필요
const RAW_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:3000';
const BASE = RAW_BASE.replace(/\/$/, '').endsWith('/api/v1')
  ? RAW_BASE.replace(/\/$/, '')
  : `${RAW_BASE.replace(/\/$/, '')}/api/v1`;

const normalizeGrid = (grid) => ({
  ...grid,
  id: grid.id ?? grid.grid_id ?? grid.gridId,
  grid_id: grid.grid_id ?? grid.id ?? grid.gridId,
  isflooded: grid.isflooded ?? grid.isFlooded ?? grid.flooded ?? grid.flood === 1,
  isFlooded: grid.isFlooded ?? grid.isflooded ?? grid.flooded ?? grid.flood === 1,
  lat: grid.lat,
  lon: grid.lon ?? grid.lng,
  lng: grid.lng ?? grid.lon,
});

const normalizeShelter = (shelter) => ({
  ...shelter,
  id: shelter.id ?? shelter.shelter_id ?? shelter.shelterId,
  shelter_id: shelter.shelter_id ?? shelter.id ?? shelter.shelterId,
  name: shelter.name ?? shelter.title ?? shelter.명칭,
  type: shelter.type ?? shelter.category ?? shelter.유형,
  address: shelter.address ?? shelter.addr ?? shelter.주소,
  lat: shelter.lat ?? shelter.latitude ?? shelter.위도,
  lon: shelter.lon ?? shelter.lng ?? shelter.longitude ?? shelter.경도,
  lng: shelter.lng ?? shelter.lon ?? shelter.longitude ?? shelter.경도,
  status: shelter.status ?? shelter.operationStatus ?? shelter.운영상태 ?? '운영중',
});

const normalizeRoute = (data) => {
  const lineFeature = data.type === 'FeatureCollection'
    ? data.features?.find((feature) => feature.geometry?.type === 'LineString')
    : null;
  const properties = lineFeature?.properties ?? data.properties ?? {};
  const geojsonWaypoints = lineFeature?.geometry?.coordinates?.map(([lon, lat]) => ({ lat, lon })) ?? null;
  const waypoints = geojsonWaypoints ?? data.waypoints ?? data.path ?? data.route ?? [];
  const totalDistance = data.totalDistance ?? data.distance ?? data.distanceMeters ?? properties.distanceM;

  return {
    ...data,
    totalMinutes: data.totalMinutes ?? data.duration ?? data.durationMinutes ?? Math.ceil((totalDistance ?? 0) / 80),
    totalDistance,
    avoidedGrids: data.avoidedGrids ?? data.avoided_grid_count ?? (properties.hasFloodedSegment ? 1 : 0),
    hasFloodedSegment: data.hasFloodedSegment ?? properties.hasFloodedSegment ?? false,
    nodeCount: data.nodeCount ?? properties.nodeCount,
    edgeCount: data.edgeCount ?? properties.edgeCount,
    waypoints: waypoints.map((point) => {
      if (Array.isArray(point)) {
        const [lon, lat] = point;
        return { lat, lon, lng: lon };
      }
      return {
        ...point,
        lat: point.lat ?? point.latitude,
        lon: point.lon ?? point.lng ?? point.longitude,
        lng: point.lng ?? point.lon ?? point.longitude,
      };
    }),
  };
};

const asArray = (value) => Array.isArray(value) ? value : [];

const request = async (method, path, params = null, body = null) => {
  const url = new URL(`${BASE}${path}`);
  if (params) {
    Object.entries(params).forEach(([k, v]) => {
      if (v !== null && v !== undefined) url.searchParams.set(k, v);
    });
  }
  const res = await fetch(url.toString(), {
    method,
    headers: { 'Content-Type': 'application/json' },
    ...(body ? { body: JSON.stringify(body) } : {}),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw Object.assign(new Error(err.message ?? `HTTP ${res.status}`), { code: err.code, status: res.status });
  }
  return res.json();
};

// ── 위치 ──────────────────────────────────────────────────────────────────
export const checkFlood = (lat, lon) =>
  request('GET', '/location/flood-check', { lat, lon });

// ── 날씨 ──────────────────────────────────────────────────────────────────
export const getWeather = (lat, lon) =>
  request('GET', '/weather/current', { lat, lon });

// ── 히트맵 ────────────────────────────────────────────────────────────────
// horizon: 'now' | '1h' | '3h' | '6h'
export const getHeatmapGrids = (lat, lon, horizon = 'now', radius = 5000) =>
  request('GET', '/heatmap/grids', { lat, lon, horizon, radius })
    .then((data) => ({
      ...data,
      grids: asArray(data.grids ?? data.items ?? data).map(normalizeGrid),
    }));

// ── 대피소 ────────────────────────────────────────────────────────────────
// type: 'all' | 'school' | 'public' | 'hotel'
export const getShelters = (lat, lon, type = 'all', radius = 3000) =>
  request('GET', '/shelters', { lat, lon, type: type === 'all' ? undefined : type, radius })
    .then((data) => ({
      ...data,
      shelters: asArray(data.shelters ?? data.items ?? data).map(normalizeShelter),
    }));

// ── 경로 ──────────────────────────────────────────────────────────────────
// mode: 'avoid_flood' | 'fastest'
export const getEvacRoute = (originLat, originLon, destLat, destLon, mode = 'avoid_flood') => {
  const body = {
    originLat,
    originLon,
    destLat,
    destLon,
    mode,
    origin: { lat: originLat, lon: originLon, lng: originLon },
    destination: { lat: destLat, lon: destLon, lng: destLon },
  };

  return request('POST', '/route/evacuation', null, body)
    .catch((error) => {
      if ([400, 404, 405, 422].includes(error.status)) {
        return request('POST', '/evacuation/route', null, body);
      }
      throw error;
    })
    .then(normalizeRoute);
};

// ── 경보 ──────────────────────────────────────────────────────────────────
export const getAlerts = (lat, lon, limit = 50) =>
  request('GET', '/alerts', { lat, lon, limit });

// ── 이메일 알림 구독 ─────────────────────────────────────────────────────
export const subscribeToGrid = (gridId, email) =>
  request('POST', '/subscriptions', null, { gridId: String(gridId), email });
