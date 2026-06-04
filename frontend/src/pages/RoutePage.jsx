import { useEffect, useRef, useState } from 'react';
import { MOCK_HEATMAP, HEATMAP_COLORS } from '../data/mockData';
import { getDirection } from '../utils/directions';
import { parseRoute } from '../utils/routeParser';

const RoutePage = ({ userLocation, shelter }) => {
  const mapRef        = useRef(null);
  const canvasRef     = useRef(null);
  const kakaoMapRef   = useRef(null);
  const featuresRef   = useRef(null);
  const floodMapRef   = useRef(new Map());
  const rafRef        = useRef(null);
  const polylinesRef  = useRef([]);
  const myMarkerRef   = useRef(null);
  const destMarkerRef = useRef(null);

  const [routeMode, setRouteMode] = useState('avoid_flood');
  const [walkTime, setWalkTime]   = useState({ normal: null, safe: null });
  const [isLoading, setIsLoading] = useState(false);
  const [routeDesc, setRouteDesc] = useState('');

  // ── Canvas 렌더링 ──────────────────────────────────────────────────────
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

      // 현재 침수 그리드만 표시
      const currentIds = floodMapRef.current;
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

  // ── 침수 격자 안에 있는지 판별 (레이 캐스팅) ─────────────────────────
  const isInsideFlood = (point) => {
    const features = featuresRef.current;
    if (!features) return false;
    for (const feature of features) {
      const grid_id = feature.properties.grid_id;
      if (!floodMapRef.current.has(grid_id)) continue;
      let coords = feature.geometry.coordinates;
      if (feature.geometry.type === 'MultiPolygon') coords = coords[0][0];
      else if (feature.geometry.type === 'Polygon') coords = coords[0];
      if (!coords?.length) continue;

      const x = point.lng, y = point.lat;
      let inside = false;
      for (let i = 0, j = coords.length - 1; i < coords.length; j = i++) {
        const xi = coords[i][0], yi = coords[i][1];
        const xj = coords[j][0], yj = coords[j][1];
        if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) inside = !inside;
      }
      if (inside) return true;
    }
    return false;
  };

  // ── 두 경로가 동일한지 비교 ───────────────────────────────────────────
  const isSamePath = (a, b) => {
    if (a.length !== b.length) return false;
    return a.every((pt, i) =>
      Math.abs(pt.lat - b[i].lat) < 0.00001 &&
      Math.abs(pt.lng - b[i].lng) < 0.00001
    );
  };

  // ── 폴리라인 렌더링 ───────────────────────────────────────────────────
  const drawRoutePolylines = (normalPath, safePath, identical) => {
    polylinesRef.current.forEach(l => l.setMap(null));
    polylinesRef.current = [];
    const kakaoMap = kakaoMapRef.current;

    if (identical) {
      const line = new window.kakao.maps.Polyline({
        path: normalPath.map(p => new window.kakao.maps.LatLng(p.lat, p.lng)),
        strokeWeight: 6, strokeColor: '#2563EB',
        strokeOpacity: 0.9, strokeStyle: 'dash',
      });
      line.setMap(kakaoMap);
      polylinesRef.current.push(line);
    } else {
      const normalLine = new window.kakao.maps.Polyline({
        path: normalPath.map(p => new window.kakao.maps.LatLng(p.lat, p.lng)),
        strokeWeight: 5, strokeColor: '#10B981',
        strokeOpacity: 0.8, strokeStyle: 'solid',
      });
      const safeLine = new window.kakao.maps.Polyline({
        path: safePath.map(p => new window.kakao.maps.LatLng(p.lat, p.lng)),
        strokeWeight: 6, strokeColor: '#2563EB',
        strokeOpacity: 0.9, strokeStyle: 'solid',
      });
      normalLine.setMap(kakaoMap);
      safeLine.setMap(kakaoMap);
      polylinesRef.current.push(normalLine, safeLine);
    }
  };

  // ── 경로 탐색 메인 로직 ───────────────────────────────────────────────
  const createRoute = async (mode) => {
    const kakaoMap = kakaoMapRef.current;
    if (!kakaoMap || !userLocation) return;

    const dest = shelter ?? { lat: 37.5172, lng: 127.0473, name: '정신여자고등학교' };
    const start = { lat: userLocation.lat, lng: userLocation.lng };
    const end   = { lat: dest.lat, lng: dest.lon ?? dest.lng };

    setIsLoading(true);

    try {
      // 마커 초기화
      if (myMarkerRef.current)    myMarkerRef.current.setMap(null);
      if (destMarkerRef.current)  destMarkerRef.current.setMap(null);

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
      const myMarker = new window.kakao.maps.CustomOverlay({
        position: new window.kakao.maps.LatLng(start.lat, start.lng),
        content: myEl, zIndex: 5,
      });
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
      const destMarker = new window.kakao.maps.CustomOverlay({
        position: new window.kakao.maps.LatLng(end.lat, end.lng),
        content: destEl, zIndex: 5,
      });
      destMarker.setMap(kakaoMap);
      destMarkerRef.current = destMarker;

      // 일반 최단경로
      const normalData = await getDirection(start, end);
      const normalPath = parseRoute(normalData);
      const normalSec  = normalData.routes[0].summary.duration;
      const normalMin  = Math.round((normalSec / 60) * 4.5);

      let safePath   = [...normalPath];
      let safeMin    = normalMin;
      let avoided    = 0;

      if (mode === 'avoid_flood') {
        // 침수 구역 충돌 지점 탐색
        let firstCollisionIdx = -1;
        for (let i = 0; i < normalPath.length; i++) {
          if (isInsideFlood(normalPath[i])) {
            firstCollisionIdx = i;
            break;
          }
        }

        if (firstCollisionIdx > 0) {
          const collisionPoint = normalPath[firstCollisionIdx - 1];
          const offsets = [
            { lat: 0.0012, lng: 0.0012 },
            { lat: 0.0012, lng: -0.0012 },
            { lat: -0.0012, lng: 0.0012 },
            { lat: -0.0012, lng: -0.0012 },
            { lat: 0.0025, lng: 0.0025 },
            { lat: 0.0025, lng: -0.0025 },
          ];

          let found = false;
          for (const offset of offsets) {
            const waypoint = {
              lat: collisionPoint.lat + offset.lat,
              lng: collisionPoint.lng + offset.lng,
            };
            try {
              const testData = await getDirection(start, end, waypoint);
              const testPath = parseRoute(testData);
              const hasFlood = testPath.some(pt => isInsideFlood(pt));
              if (!hasFlood && testPath.length > 0) {
                safePath = testPath;
                safeMin  = Math.round((testData.routes[0].summary.duration / 60) * 4.5);
                avoided  = 1;
                found    = true;
                break;
              }
            } catch { continue; }
          }

          if (!found) {
            const backup = {
              lat: collisionPoint.lat + 0.0035,
              lng: collisionPoint.lng + 0.0035,
            };
            const backupData = await getDirection(start, end, backup);
            safePath = parseRoute(backupData);
            safeMin  = Math.round((backupData.routes[0].summary.duration / 60) * 4.5);
            avoided  = 1;
          }
        }
      }

      const identical = isSamePath(normalPath, safePath);
      drawRoutePolylines(normalPath, safePath, identical);

      setWalkTime({ normal: normalMin, safe: safeMin });

      if (mode === 'avoid_flood') {
        setRouteDesc(avoided > 0
          ? `침수구역을 우회하는 안전한 경로로 안내해요`
          : `현재 경로에 침수구역이 없어요`
        );
      } else {
        setRouteDesc('최단 경로로 안내해요');
      }

      // 경로가 보이도록 지도 범위 조정
      kakaoMap.setBounds(
        new window.kakao.maps.LatLngBounds(
          new window.kakao.maps.LatLng(start.lat, start.lng),
          new window.kakao.maps.LatLng(end.lat, end.lng)
        )
      );
    } catch (e) {
      console.error('[RoutePage]', e);
    } finally {
      setIsLoading(false);
    }
  };

  // ── 카카오맵 초기화 ───────────────────────────────────────────────────
  useEffect(() => {
    const wait = setInterval(() => {
      if (window.kakao && window.kakao.maps) {
        clearInterval(wait);
        const kakaoMap = new window.kakao.maps.Map(mapRef.current, {
          center: new window.kakao.maps.LatLng(userLocation.lat, userLocation.lng),
          level: 6,
        });
        kakaoMapRef.current = kakaoMap;

        window.kakao.maps.event.addListener(kakaoMap, 'idle',         redraw);
        window.kakao.maps.event.addListener(kakaoMap, 'zoom_changed', redraw);
        window.addEventListener('resize', redraw);

        fetch('/seoul_grid.geojson')
          .then(r => r.json())
          .then(g => {
            featuresRef.current = g.features;
            // 현재 침수 격자 세팅
            floodMapRef.current = new Set(MOCK_HEATMAP.now);
            redraw();
            createRoute('avoid_flood');
          });
      }
    }, 100);
    return () => clearInterval(wait);
  }, []);

  const handleModeChange = (mode) => {
    setRouteMode(mode);
    createRoute(mode);
  };

  const dest = shelter ?? { name: '정신여자고등학교' };

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
        display: 'flex', alignItems: 'center', gap: 10,
      }}>
        <div style={{
          width: 36, height: 36, borderRadius: 10,
          background: '#EFF6FF', display: 'flex',
          alignItems: 'center', justifyContent: 'center', fontSize: 18,
        }}>🧭</div>
        <div>
          <div style={{ fontSize: 11, color: '#9CA3AF' }}>도착지</div>
          <div style={{ fontSize: 14, fontWeight: 700, color: '#111' }}>{dest.name}</div>
        </div>
      </div>

      {/* 경로 모드 선택 */}
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

      {/* 하단 경로 정보 */}
      <div style={{
        position: 'absolute', bottom: 0, left: 0, right: 0,
        background: 'white', borderRadius: '20px 20px 0 0',
        boxShadow: '0 -4px 20px rgba(0,0,0,0.1)', zIndex: 10,
        padding: '16px 20px 40px',
      }}>
        <div style={{ width: 36, height: 4, background: '#E5E7EB', borderRadius: 2, margin: '0 auto 16px' }} />

        {isLoading ? (
          <div style={{ textAlign: 'center', padding: '20px 0', color: '#9CA3AF' }}>
            경로 탐색 중…
          </div>
        ) : walkTime.normal ? (
          <>
            <div style={{ display: 'flex', gap: 14, marginBottom: 16 }}>
              {/* 안전우선 모드일 때 두 경로 시간 비교 표시 */}
              {routeMode === 'avoid_flood' && walkTime.normal !== walkTime.safe ? (
                <>
                  <div style={{
                    flex: 1, background: '#F0FDF4', borderRadius: 12, padding: '12px',
                    textAlign: 'center',
                  }}>
                    <div style={{ fontSize: 10, color: '#6B7280', marginBottom: 2 }}>일반 경로</div>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4 }}>
                      <span style={{ width: 12, height: 4, background: '#10B981', display: 'inline-block', borderRadius: 2 }} />
                    </div>
                    <div style={{ fontSize: 22, fontWeight: 800, color: '#111' }}>{walkTime.normal}<span style={{ fontSize: 12, color: '#6B7280' }}>분</span></div>
                  </div>
                  <div style={{
                    flex: 1, background: '#EFF6FF', borderRadius: 12, padding: '12px',
                    textAlign: 'center', border: '2px solid #DBEAFE',
                  }}>
                    <div style={{ fontSize: 10, color: '#3B82F6', fontWeight: 700, marginBottom: 2 }}>침수 우회</div>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4 }}>
                      <span style={{ width: 12, height: 4, background: '#2563EB', display: 'inline-block', borderRadius: 2 }} />
                    </div>
                    <div style={{ fontSize: 22, fontWeight: 800, color: '#2563EB' }}>{walkTime.safe}<span style={{ fontSize: 12, color: '#6B7280' }}>분</span></div>
                  </div>
                </>
              ) : (
                <div style={{
                  flex: 1, background: '#F0FDF4', borderRadius: 12, padding: '12px 16px',
                  display: 'flex', alignItems: 'center', gap: 14,
                }}>
                  <div style={{ textAlign: 'center', minWidth: 60 }}>
                    <div style={{ fontSize: 11, color: '#6B7280', marginBottom: 2 }}>예상시간</div>
                    <div style={{ fontSize: 24, fontWeight: 800, color: '#111', lineHeight: 1 }}>
                      {routeMode === 'avoid_flood' ? walkTime.safe : walkTime.normal}
                      <span style={{ fontSize: 12, color: '#6B7280' }}>분</span>
                    </div>
                  </div>
                  <div style={{ fontSize: 13, color: '#374151', lineHeight: 1.5 }}>
                    {routeDesc || (routeMode === 'fastest' ? '최단 경로로 안내해요' : '안전한 경로로 안내해요')}
                  </div>
                </div>
              )}
            </div>

            {routeMode === 'avoid_flood' && walkTime.normal === walkTime.safe && (
              <div style={{
                fontSize: 12, color: '#2563EB', background: '#EFF6FF',
                padding: '8px 12px', borderRadius: 8, textAlign: 'center', marginBottom: 12,
              }}>
                💡 두 경로가 동일하여 파란색 점선으로 표시됩니다
              </div>
            )}

            <button style={{
              width: '100%', padding: '15px', background: '#3B82F6', color: 'white',
              border: 'none', borderRadius: 14, fontSize: 16, fontWeight: 700, cursor: 'pointer',
            }}>🚶 길안내 시작</button>
          </>
        ) : null}
      </div>
    </div>
  );
};

export default RoutePage;
