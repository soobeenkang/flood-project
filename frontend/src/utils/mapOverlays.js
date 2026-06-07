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
