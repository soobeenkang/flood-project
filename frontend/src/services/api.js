// Base URL 환경변수로 관리
// .env 파일에 VITE_API_BASE_URL=http://localhost:3000 추가 필요
const BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:3000';

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
  request('GET', '/heatmap/grids', { lat, lon, horizon, radius });

// ── 대피소 ────────────────────────────────────────────────────────────────
// type: 'all' | 'school' | 'public' | 'hotel'
export const getShelters = (lat, lon, type = 'all', radius = 3000) =>
  request('GET', '/shelters', { lat, lon, type, radius });

// ── 경로 ──────────────────────────────────────────────────────────────────
// mode: 'avoid_flood' | 'fastest'
export const getEvacRoute = (originLat, originLon, destLat, destLon, mode = 'avoid_flood') =>
  request('POST', '/route/evacuation', null, {
    originLat, originLon, destLat, destLon, mode,
  });

// ── 경보 ──────────────────────────────────────────────────────────────────
export const getAlerts = (lat, lon, limit = 50) =>
  request('GET', '/alerts', { lat, lon, limit });
