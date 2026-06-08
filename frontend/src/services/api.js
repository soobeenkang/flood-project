// Base URL 환경변수로 관리
// .env 파일에 VITE_API_BASE_URL=http://localhost:3000 추가 필요
const RAW_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:3000';
const BASE = RAW_BASE.replace(/\/$/, '').endsWith('/api/v1')
  ? RAW_BASE.replace(/\/$/, '')
  : `${RAW_BASE.replace(/\/$/, '')}/api/v1`;

const toBool = (value) => value === true || value === 1 || value === '1' || value === 'true';

const normalizeGrid = (grid) => {
  const gridId = grid.grid_id ?? grid.id ?? grid.gridId;
  const isFlooded = toBool(grid.isFlooded ?? grid.isflooded ?? grid.is_flooded ?? grid.flooded ?? grid.flood);

  return {
    ...grid,
    id: String(gridId),
    grid_id: String(gridId),
    isflooded: isFlooded,
    isFlooded,
    lat: grid.lat,
    lon: grid.lon ?? grid.lng,
    lng: grid.lng ?? grid.lon,
  };
};

const normalizeShelterType = (type, shelter = {}) => {
  const raw = [
    type,
    shelter.name,
    shelter.address,
    shelter.title,
    shelter.addr,
    shelter.명칭,
    shelter.주소,
  ].map((value) => String(value ?? '').trim().toLowerCase()).join(' ');

  if (!raw) return 'public';
  if (['school', '학교', '초등학교', '중학교', '고등학교', '대학교'].some((value) => raw.includes(value))) {
    return 'school';
  }
  if (['hotel', '호텔', '숙박', '모텔'].some((value) => raw.includes(value))) {
    return 'hotel';
  }
  if (['public', '공공', '기관', '구청', '주민센터', '센터', '체육관', '복지관'].some((value) => raw.includes(value))) {
    return 'public';
  }
  return raw;
};

const normalizeShelter = (shelter) => ({
  ...shelter,
  id: shelter.id ?? shelter.shelter_id ?? shelter.shelterId,
  shelter_id: shelter.shelter_id ?? shelter.id ?? shelter.shelterId,
  name: shelter.name ?? shelter.title ?? shelter.명칭,
  type: normalizeShelterType(shelter.type ?? shelter.category ?? shelter.유형, shelter),
  rawType: shelter.type ?? shelter.category ?? shelter.유형,
  address: shelter.address ?? shelter.addr ?? shelter.주소,
  lat: shelter.lat ?? shelter.latitude ?? shelter.위도,
  lon: shelter.lon ?? shelter.lng ?? shelter.longitude ?? shelter.경도,
  lng: shelter.lng ?? shelter.lon ?? shelter.longitude ?? shelter.경도,
  status: shelter.status ?? shelter.operationStatus ?? shelter.운영상태 ?? '운영중',
});

const normalizeRouteFeature = (feature) => {
  const properties = feature?.properties ?? {};
  const coordinates = feature?.geometry?.coordinates ?? [];
  const totalDistance = properties.distanceM;

  return {
    id: feature?.id,
    routeType: properties.routeType ?? feature?.id,
    totalMinutes: Math.ceil((totalDistance ?? 0) / 80),
    totalDistance,
    avoidedGrids: properties.bypassedCount ?? properties.avoidedGrids ?? properties.avoided_grid_count ?? 0,
    hasFloodedSegment: properties.hasFloodedSegment ?? false,
    nodeCount: properties.nodeCount,
    edgeCount: properties.edgeCount,
    waypoints: coordinates.map(([lon, lat]) => ({ lat, lon, lng: lon })),
  };
};

const normalizeWaypoint = (point) => {
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
};

const normalizeRoute = (data, mode = 'avoid_flood') => {
  if (data.type === 'FeatureCollection') {
    const routeFeatures = asArray(data.features)
      .filter((feature) => feature.geometry?.type === 'LineString')
      .map(normalizeRouteFeature);
    const safeRoute = routeFeatures.find((route) => route.id === 'safe_route' || route.routeType === 'safe');
    const normalRoute = routeFeatures.find((route) => route.id === 'normal_route' || route.routeType === 'normal');
    const selectedRoute = mode === 'fastest'
      ? (normalRoute ?? safeRoute ?? routeFeatures[0])
      : (safeRoute ?? normalRoute ?? routeFeatures[0]);

    return {
      ...data,
      routes: {
        safe: safeRoute,
        normal: normalRoute,
      },
      selectedRoute,
      ...(selectedRoute ?? {}),
    };
  }

  const properties = data.properties ?? {};
  const waypoints = data.waypoints ?? data.path ?? data.route ?? [];
  const totalDistance = data.totalDistance ?? data.distance ?? data.distanceMeters ?? properties.distanceM;

  return {
    ...data,
    totalMinutes: data.totalMinutes ?? data.duration ?? data.durationMinutes ?? Math.ceil((totalDistance ?? 0) / 80),
    totalDistance,
    avoidedGrids: data.bypassedCount ?? data.avoidedGrids ?? data.avoided_grid_count ?? properties.bypassedCount ?? 0,
    hasFloodedSegment: data.hasFloodedSegment ?? properties.hasFloodedSegment ?? false,
    nodeCount: data.nodeCount ?? properties.nodeCount,
    edgeCount: data.edgeCount ?? properties.edgeCount,
    waypoints: waypoints.map(normalizeWaypoint),
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
  request('GET', '/shelters', { lat, lon, radius })
    .then((data) => ({
      ...data,
      shelters: asArray(data.shelters ?? data.items ?? data)
        .map(normalizeShelter)
        .filter((shelter) => type === 'all' || shelter.type === type),
    }));

// ── 경로 ──────────────────────────────────────────────────────────────────
// mode: 'avoid_flood' | 'fastest'
export const getEvacRoute = (originLat, originLon, destLat, destLon, mode = 'avoid_flood') => {
  const params = { startLat: originLat, startLon: originLon, endLat: destLat, endLon: destLon, mode };
  return request('GET', '/route/evacuation', params).then((data) => normalizeRoute(data, mode));
};

// ── 경보 ──────────────────────────────────────────────────────────────────
export const getAlerts = (lat, lon, limit = 50) =>
  request('GET', '/alerts', { lat, lon, limit });

// ── 이메일 알림 구독 ─────────────────────────────────────────────────────
export const subscribeToGrid = (gridId, email) =>
  request('POST', '/subscriptions', null, { gridId: String(gridId), email });