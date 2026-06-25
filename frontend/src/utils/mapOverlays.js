export const getLocationCoords = (location) => ({
  lat: location?.lat,
  lng: location?.lng ?? location?.lon,
});

export const createCurrentLocationOverlay = (kakao, location) => {
  const { lat, lng } = getLocationCoords(location);
  if (lat === undefined || lng === undefined) return null;

  const el = document.createElement('div');
  el.style.cssText = `
    width: 26px;
    height: 26px;
    border-radius: 50%;
    background: rgba(59, 130, 246, 0.18);
    border: 1px solid rgba(59, 130, 246, 0.45);
    display: flex;
    align-items: center;
    justify-content: center;
    box-sizing: border-box;
  `;

  const dot = document.createElement('div');
  dot.style.cssText = `
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: #2563EB;
    border: 3px solid white;
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.35);
    box-sizing: border-box;
  `;
  el.appendChild(dot);

  return new kakao.maps.CustomOverlay({
    position: new kakao.maps.LatLng(lat, lng),
    content: el,
    zIndex: 12,
  });
};

export const createSearchLocationOverlay = (kakao, location) => {
  const { lat, lng } = getLocationCoords(location);
  if (lat === undefined || lng === undefined) return null;

  const el = document.createElement('div');
  el.style.cssText = `
    width: 30px;
    height: 30px;
    border-radius: 50% 50% 50% 0;
    background: #EF4444;
    border: 3px solid white;
    transform: rotate(-45deg);
    box-shadow: 0 2px 10px rgba(0,0,0,0.22);
    box-sizing: border-box;
  `;

  const inner = document.createElement('div');
  inner.style.cssText = `
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: white;
    margin: 7px auto 0;
  `;
  el.appendChild(inner);

  return new kakao.maps.CustomOverlay({
    position: new kakao.maps.LatLng(lat, lng),
    content: el,
    zIndex: 11,
  });
};

export const GRID_HALF_LAT = 0.00045;
export const GRID_HALF_LON = 0.00057;

export const getGridCoords = (grid) => {
  const lat = grid.lat;
  const lng = grid.lon ?? grid.lng;
  if (lat === undefined || lng === undefined) return null;

  return [
    [lng - GRID_HALF_LON, lat - GRID_HALF_LAT],
    [lng + GRID_HALF_LON, lat - GRID_HALF_LAT],
    [lng + GRID_HALF_LON, lat + GRID_HALF_LAT],
    [lng - GRID_HALF_LON, lat + GRID_HALF_LAT],
  ];
};

const getDistanceMeters = (a, b) => {
  const toRad = (value) => value * Math.PI / 180;
  const earthRadius = 6371000;
  const dLat = toRad(b.lat - a.lat);
  const dLng = toRad(b.lng - a.lng);
  const lat1 = toRad(a.lat);
  const lat2 = toRad(b.lat);
  const h = Math.sin(dLat / 2) ** 2
    + Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLng / 2) ** 2;

  return earthRadius * 2 * Math.atan2(Math.sqrt(h), Math.sqrt(1 - h));
};

export const getVisibleRequestArea = (kakaoMap) => {
  const center = kakaoMap.getCenter();
  const bounds = kakaoMap.getBounds();
  const sw = bounds.getSouthWest();
  const ne = bounds.getNorthEast();
  const centerPoint = { lat: center.getLat(), lng: center.getLng() };
  const corners = [
    { lat: sw.getLat(), lng: sw.getLng() },
    { lat: sw.getLat(), lng: ne.getLng() },
    { lat: ne.getLat(), lng: sw.getLng() },
    { lat: ne.getLat(), lng: ne.getLng() },
  ];
  const radius = Math.ceil(Math.max(...corners.map((corner) => getDistanceMeters(centerPoint, corner))) * 1.15);

  return {
    lat: centerPoint.lat,
    lng: centerPoint.lng,
    radius: Math.min(Math.max(radius, 1000), 10000),
  };
};
