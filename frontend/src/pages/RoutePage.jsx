import { useEffect, useRef, useState } from 'react';
import { HEATMAP_COLORS, SHELTER_TYPES } from '../data/mockData';
import { getEvacRoute, getHeatmapGrids, getShelters } from '../services/api';

const RoutePage = ({ userLocation, shelter }) => {
  const mapRef        = useRef(null);
  const canvasRef     = useRef(null);
  const kakaoMapRef   = useRef(null);
  const featuresRef   = useRef(null);
  const floodIdsRef   = useRef(new Set());
  const rafRef        = useRef(null);
  const polylineRef   = useRef(null);
  const myMarkerRef   = useRef(null);
  const destMarkerRef = useRef(null);
  const shelterMarkersRef = useRef([]);

  const [routeMode, setRouteMode] = useState('avoid_flood');
  const [routeInfo, setRouteInfo] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isShelterLoading, setIsShelterLoading] = useState(false);
  const [routeError, setRouteError] = useState(null);
  const [shelters, setShelters] = useState([]);
  const [selectedShelter, setSelectedShelter] = useState(shelter ?? null);

  const fetchShelters = async () => {
    setIsShelterLoading(true);
    try {
      const data = await getShelters(userLocation.lat, userLocation.lng, 'all', 3000);
      setShelters(data.shelters ?? []);
    } catch (e) {
      console.error('[RoutePage] 대피소 fetch 실패:', e);
      setShelters([]);
    } finally {
      setIsShelterLoading(false);
    }
  };

  const fetchCurrentHeatmap = async () => {
    try {
      const data = await getHeatmapGrids(userLocation.lat, userLocation.lng, 'now', 5000);
      floodIdsRef.current = new Set(
        (data.grids ?? []).filter(g => g.isFlooded).map(g => g.grid_id)
      );
    } catch (e) {
      console.error('[RoutePage] 히트맵 fetch 실패:', e);
      floodIdsRef.current = new Set();
    }
  };

  const redraw = () => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      const canvas   = canvasRef.current;
      const kakaoMap = kakaoMapRef.current;
      const features = featuresRef.current;
      if (!canvas || !kakaoMap || !features) return;

      canvas.width  = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;

      const ctx    = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const bounds = kakaoMap.getBounds();
      const sw     = bounds.getSouthWest();
      const ne     = bounds.getNorthEast();
      const W      = canvas.width;
      const H      = canvas.height;

      const lngToX = (lng) => (lng - sw.getLng()) / (ne.getLng() - sw.getLng()) * W;
      const latToY = (lat) => (1 - (lat - sw.getLat()) / (ne.getLat() - sw.getLat())) * H;

      const currentIds = floodIdsRef.current;
      features.forEach((feature) => {
        const { grid_id, lon, lat } = feature.properties;
        if (!currentIds.has(grid_id)) return;
        if (lon < sw.getLng() || lon > ne.getLng() ||
            lat < sw.getLat() || lat > ne.getLat()) return;

        const coords = feature.geometry.coordinates[0];
        ctx.beginPath();
        coords.forEach(([lng, la], i) => {
          const x = lngToX(lng);
          const y = latToY(la);
          if (i === 0) ctx.moveTo(x, y);
          else         ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.fillStyle = HEATMAP_COLORS.now;
        ctx.fill();
      });
    });
  };

  const drawRoute = (kakaoMap, waypoints) => {
    if (polylineRef.current)    polylineRef.current.setMap(null);
    if (myMarkerRef.current)    myMarkerRef.current.setMap(null);
    if (destMarkerRef.current)  destMarkerRef.current.setMap(null);

    if (!selectedShelter) return;
    const dest = selectedShelter;
    const destLat = dest.lat;
    const destLon = dest.lon ?? dest.lng;
    if (destLat === undefined || destLon === undefined) {
      setRouteInfo(null);
      setRouteError('선택한 대피소의 좌표 정보가 없습니다.');
      return;
    }

    const origin  = new window.kakao.maps.LatLng(userLocation.lat, userLocation.lng);
    const destPos = new window.kakao.maps.LatLng(destLat, destLon);

    // 경로 좌표 (API 응답 waypoints 또는 직선)
    const path = waypoints && waypoints.length > 0
      ? waypoints.map(w => new window.kakao.maps.LatLng(w.lat, w.lon ?? w.lng))
      : [origin, destPos];

    const polyline = new window.kakao.maps.Polyline({
      path,
      strokeWeight: 5,
      strokeColor: '#3B82F6',
      strokeOpacity: 1,
      strokeStyle: 'solid',
    });
    polyline.setMap(kakaoMap);
    polylineRef.current = polyline;

    // 내 위치 마커
    const myEl = document.createElement('div');
    myEl.style.cssText = `
      width:20px;height:20px;background:white;
      border:3px solid #EF4444;border-radius:50%;
      display:flex;align-items:center;justify-content:center;
    `;
    myEl.appendChild(Object.assign(document.createElement('div'), {
      style: 'width:8px;height:8px;background:#EF4444;border-radius:50%;',
    }));
    const myMarker = new window.kakao.maps.CustomOverlay({ position: origin, content: myEl, zIndex: 5 });
    myMarker.setMap(kakaoMap);
    myMarkerRef.current = myMarker;

    // 목적지 마커
    const destEl = document.createElement('div');
    destEl.style.cssText = `
      width:36px;height:36px;background:#3B82F6;
      border:3px solid white;border-radius:50%;
      display:flex;align-items:center;justify-content:center;
      font-size:18px;box-shadow:0 2px 8px rgba(0,0,0,0.2);
    `;
    destEl.innerHTML = '🏠';
    const dm = new window.kakao.maps.CustomOverlay({ position: destPos, content: destEl, zIndex: 5 });
    dm.setMap(kakaoMap);
    destMarkerRef.current = dm;

    kakaoMap.setBounds(new window.kakao.maps.LatLngBounds(origin, destPos));
  };

  const addShelterMarkers = (kakaoMap) => {
    shelterMarkersRef.current.forEach(marker => marker.setMap(null));
    shelterMarkersRef.current = [];

    shelters.forEach((item) => {
      const lat = item.lat;
      const lon = item.lon ?? item.lng;
      if (lat === undefined || lon === undefined) return;

      const isSelected = selectedShelter?.id === item.id;
      const el = document.createElement('button');
      el.type = 'button';
      el.style.cssText = `
        width:${isSelected ? 46 : 38}px;height:${isSelected ? 46 : 38}px;
        border-radius:50%;border:3px solid ${isSelected ? '#2563EB' : '#E5E7EB'};
        background:${isSelected ? '#3B82F6' : 'white'};
        display:flex;align-items:center;justify-content:center;
        font-size:${isSelected ? 22 : 18}px;cursor:pointer;
        box-shadow:0 2px 8px rgba(0,0,0,0.16);
      `;
      el.innerHTML = SHELTER_TYPES[item.type]?.emoji ?? '🏢';
      el.addEventListener('click', () => setSelectedShelter(item));

      const marker = new window.kakao.maps.CustomOverlay({
        position: new window.kakao.maps.LatLng(lat, lon),
        content: el,
        zIndex: isSelected ? 8 : 6,
      });
      marker.setMap(kakaoMap);
      shelterMarkersRef.current.push(marker);
    });
  };

  const fetchRoute = async (mode, destination = selectedShelter) => {
    if (!destination) {
      setRouteInfo(null);
      setRouteError('지도나 목록에서 대피소를 선택해주세요.');
      return;
    }

    setIsLoading(true);
    setRouteError(null);
    try {
      const dest = destination;
      const destLat = dest.lat;
      const destLon = dest.lon ?? dest.lng;
      if (destLat === undefined || destLon === undefined) {
        setRouteInfo(null);
        setRouteError('선택한 대피소의 좌표 정보가 없습니다.');
        return;
      }

      const data = await getEvacRoute(
        userLocation.lat, userLocation.lng,
        destLat, destLon,
        mode
      );
      setRouteInfo({
        totalMinutes:  data.totalMinutes ?? 0,
        totalDistance: data.totalDistance ?? 0,
        avoidedGrids:  data.avoidedGrids,
        desc: mode === 'avoid_flood'
          ? `침수구역 ${data.avoidedGrids}곳을 우회해요`
          : '최단 경로로 안내해요',
        detail: mode === 'avoid_flood'
          ? '침수 위험 구역을 피해 안전한 경로로 안내합니다.'
          : '침수구역을 통과할 수 있습니다. 주의하세요.',
        waypoints: data.waypoints ?? [],
      });
      if (kakaoMapRef.current) {
        try {
          drawRoute(kakaoMapRef.current, data.waypoints ?? []);
        } catch (drawError) {
          console.error('[RoutePage drawRoute]', drawError);
        }
      }
    } catch (e) {
      console.error('[RoutePage]', e);
      setRouteInfo(null);
      setRouteError('경로 정보를 불러오지 못했습니다.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    const wait = setInterval(() => {
      if (window.kakao && window.kakao.maps) {
        clearInterval(wait);
        const kakaoMap = new window.kakao.maps.Map(mapRef.current, {
          center: new window.kakao.maps.LatLng(userLocation.lat, userLocation.lng),
          level: 6,
        });
        kakaoMapRef.current = kakaoMap;

        window.kakao.maps.event.addListener(kakaoMap, 'center_changed', redraw);
        window.kakao.maps.event.addListener(kakaoMap, 'zoom_changed',   redraw);
        window.kakao.maps.event.addListener(kakaoMap, 'dragend',        redraw);
        window.kakao.maps.event.addListener(kakaoMap, 'tilesloaded',    redraw);

        Promise.all([
          fetch('/seoul_grid.geojson').then(r => r.json()),
          fetchCurrentHeatmap(),
          fetchShelters(),
        ]).then(([geojson]) => {
          featuresRef.current = geojson.features;
          redraw();
          if (selectedShelter) fetchRoute(routeMode, selectedShelter);
        }).catch(e => console.error('[RoutePage init]', e));
      }
    }, 100);
    return () => clearInterval(wait);
  }, []);

  useEffect(() => {
    if (shelter) setSelectedShelter(shelter);
  }, [shelter]);

  useEffect(() => {
    if (kakaoMapRef.current) addShelterMarkers(kakaoMapRef.current);
  }, [shelters, selectedShelter]);

  useEffect(() => {
    if (selectedShelter) {
      fetchRoute(routeMode, selectedShelter);
    } else {
      setRouteInfo(null);
      setRouteError(null);
    }
  }, [selectedShelter]);

  const handleModeChange = (mode) => {
    setRouteMode(mode);
    fetchRoute(mode, selectedShelter);
  };

  const dest = selectedShelter;

  return (
    <div style={{ position: 'relative', height: '100%' }}>
      <div style={{ position: 'absolute', inset: 0 }}>
        <div ref={mapRef} style={{ width: '100%', height: '100%' }} />
        <canvas ref={canvasRef} style={{
          position: 'absolute', top: 0, left: 0,
          width: '100%', height: '100%',
          pointerEvents: 'none', zIndex: 3,
        }} />
      </div>

      {/* 상단 도착지 */}
      <div style={{
        position: 'absolute', top: 12, left: 12, right: 12,
        background: 'white', borderRadius: 16, padding: '12px 16px',
        boxShadow: '0 2px 12px rgba(0,0,0,0.12)', zIndex: 10,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 10,
            background: '#EFF6FF', display: 'flex',
            alignItems: 'center', justifyContent: 'center', fontSize: 18,
          }}>🧭</div>
          <div>
            <div style={{ fontSize: 11, color: '#9CA3AF' }}>도착지</div>
            <div style={{ fontSize: 14, fontWeight: 700, color: '#111' }}>
              {dest?.name ?? '목적지 미선택'} {routeInfo ? `· ${(routeInfo.totalDistance / 1000).toFixed(1)}km` : ''}
            </div>
          </div>
        </div>
      </div>

      {/* 경로 모드 */}
      <div style={{
        position: 'absolute', top: 76, left: 12, right: 12,
        display: 'flex', gap: 8, zIndex: 10,
      }}>
        {[
          { mode: 'avoid_flood', label: '🛡️ 안전 우선', badge: '추천' },
          { mode: 'fastest',     label: '⚡ 최단 시간' },
        ].map((item) => (
          <button key={item.mode} onClick={() => handleModeChange(item.mode)} style={{
            flex: 1, padding: '10px', border: 'none', borderRadius: 12, cursor: 'pointer',
            fontSize: 13, fontWeight: 700,
            background: routeMode === item.mode ? 'white' : 'rgba(255,255,255,0.6)',
            color: routeMode === item.mode ? '#3B82F6' : '#9CA3AF',
            boxShadow: routeMode === item.mode ? '0 2px 8px rgba(0,0,0,0.12)' : 'none',
          }}>
            {item.label}
            {item.badge && routeMode === item.mode && (
              <span style={{
                marginLeft: 6, background: '#DBEAFE', color: '#1D4ED8',
                fontSize: 10, padding: '2px 6px', borderRadius: 10,
              }}>{item.badge}</span>
            )}
          </button>
        ))}
      </div>

      {/* 하단 정보 */}
      <div style={{
        position: 'absolute', bottom: 0, left: 0, right: 0,
        background: 'white', borderRadius: '20px 20px 0 0',
        boxShadow: '0 -4px 20px rgba(0,0,0,0.1)', zIndex: 10,
        padding: '16px 20px 40px',
      }}>
        <div style={{ width: 36, height: 4, background: '#E5E7EB', borderRadius: 2, margin: '0 auto 16px' }} />

        <div style={{ marginBottom: 14 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <div style={{ fontSize: 14, fontWeight: 700, color: '#111' }}>대피소 선택</div>
            <div style={{ fontSize: 12, color: '#9CA3AF' }}>
              {isShelterLoading ? '불러오는 중…' : `${shelters.length}곳`}
            </div>
          </div>
          <div style={{
            display: 'flex', gap: 8, overflowX: 'auto',
            paddingBottom: 2, scrollbarWidth: 'none',
          }}>
            {shelters.length === 0 && !isShelterLoading ? (
              <div style={{ fontSize: 13, color: '#9CA3AF', padding: '8px 0' }}>
                표시할 대피소가 없어요
              </div>
            ) : shelters.map((item) => {
              const isSelected = selectedShelter?.id === item.id;
              return (
                <button
                  key={item.id ?? item.name}
                  onClick={() => setSelectedShelter(item)}
                  style={{
                    flex: '0 0 auto', maxWidth: 180, padding: '9px 12px',
                    border: `1.5px solid ${isSelected ? '#3B82F6' : '#E5E7EB'}`,
                    borderRadius: 12, background: isSelected ? '#EFF6FF' : 'white',
                    color: '#111', cursor: 'pointer', textAlign: 'left',
                  }}
                >
                  <div style={{ fontSize: 13, fontWeight: 700, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {(SHELTER_TYPES[item.type]?.emoji ?? '🏢')} {item.name}
                  </div>
                  <div style={{ fontSize: 11, color: '#6B7280', marginTop: 3 }}>
                    {item.distance !== undefined ? `${item.distance}m` : item.status ?? '대피소'}
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {isLoading ? (
          <div style={{ textAlign: 'center', padding: '20px 0', color: '#9CA3AF' }}>경로 탐색 중…</div>
        ) : routeError ? (
          <div style={{ textAlign: 'center', padding: '20px 0', color: '#6B7280', fontSize: 14 }}>
            {routeError}
          </div>
        ) : routeInfo && (
          <>
            <div style={{ display: 'flex', gap: 14, marginBottom: 16 }}>
              <div style={{
                background: '#F0FDF4', borderRadius: 12, padding: '12px 16px',
                minWidth: 80, textAlign: 'center',
              }}>
                <div style={{ fontSize: 11, color: '#6B7280', marginBottom: 2 }}>예상시간</div>
                <div style={{ fontSize: 24, fontWeight: 800, color: '#111', lineHeight: 1 }}>
                  {routeInfo.totalMinutes}
                </div>
                <div style={{ fontSize: 11, color: '#6B7280' }}>분</div>
                <div style={{ fontSize: 11, color: '#9CA3AF', marginTop: 4 }}>
                  {(routeInfo.totalDistance / 1000).toFixed(1)}km
                </div>
              </div>
              <div style={{ flex: 1, display: 'flex', alignItems: 'center' }}>
                <div>
                  <div style={{ fontSize: 14, fontWeight: 700, color: '#111', marginBottom: 4 }}>
                    ✅ {routeInfo.desc}
                  </div>
                  <div style={{ fontSize: 12, color: '#6B7280', lineHeight: 1.5 }}>
                    {routeInfo.detail}
                  </div>
                </div>
              </div>
            </div>
            <button style={{
              width: '100%', padding: '15px', background: '#3B82F6', color: 'white',
              border: 'none', borderRadius: 14, fontSize: 16, fontWeight: 700, cursor: 'pointer',
            }}>🚶 길안내 시작</button>
          </>
        )}
      </div>
    </div>
  );
};

export default RoutePage;
