import { useEffect, useRef, useState } from 'react';
import { MOCK_HEATMAP, HEATMAP_COLORS } from '../data/mockData';

const RoutePage = ({ userLocation, shelter }) => {
  const mapRef      = useRef(null);
  const canvasRef   = useRef(null);
  const kakaoMapRef = useRef(null);
  const featuresRef = useRef(null);
  const rafRef      = useRef(null);
  const polylineRef = useRef(null);
  const myMarkerRef = useRef(null);
  const destMarkerRef = useRef(null);

  const [routeMode, setRouteMode] = useState('safe'); // 'safe' | 'fast'
  const [routeInfo] = useState({
    safe: { duration: 34, distance: 9.1, desc: '침수구역 3곳을 우회해요', detail: '침수 위험 구역을 피해 안전한 경로로 안내합니다.' },
    fast: { duration: 20, distance: 8.4, desc: '최단 경로로 안내해요', detail: '침수구역을 통과할 수 있습니다. 주의하세요.' },
  });

  const drawCanvas = () => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      const canvas   = canvasRef.current;
      const kakaoMap = kakaoMapRef.current;
      const features = featuresRef.current;
      if (!canvas || !kakaoMap || !features) return;

      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const bounds = kakaoMap.getBounds();
      const sw     = bounds.getSouthWest();
      const ne     = bounds.getNorthEast();
      const proj   = kakaoMap.getProjection();

      // 경로 화면에서는 현재 침수 그리드만 표시
      const currentIds = new Set(MOCK_HEATMAP.current);
      features.forEach((feature) => {
        const { grid_id, lon, lat } = feature.properties;
        if (!currentIds.has(grid_id)) return;
        if (lon < sw.getLng() || lon > ne.getLng() ||
            lat < sw.getLat() || lat > ne.getLat()) return;

        const coords = feature.geometry.coordinates[0];
        ctx.beginPath();
        coords.forEach(([lng, la], i) => {
          const pt = proj.pointFromCoords(new window.kakao.maps.LatLng(la, lng));
          if (i === 0) ctx.moveTo(pt.x, pt.y);
          else         ctx.lineTo(pt.x, pt.y);
        });
        ctx.closePath();
        ctx.fillStyle = HEATMAP_COLORS.current;
        ctx.fill();
      });
    });
  };

  const drawRoute = (kakaoMap) => {
    if (polylineRef.current)    polylineRef.current.setMap(null);
    if (myMarkerRef.current)    myMarkerRef.current.setMap(null);
    if (destMarkerRef.current)  destMarkerRef.current.setMap(null);

    const dest = shelter ?? { lat: 37.5172, lng: 127.0473, name: '정신여자고등학교' };

    const origin  = new window.kakao.maps.LatLng(userLocation.lat, userLocation.lng);
    const destPos = new window.kakao.maps.LatLng(dest.lat, dest.lng);

    // 경로선
    const polyline = new window.kakao.maps.Polyline({
      path: [origin, destPos],
      strokeWeight: 5,
      strokeColor: '#3B82F6',
      strokeOpacity: 1,
      strokeStyle: 'solid',
    });
    polyline.setMap(kakaoMap);
    polylineRef.current = polyline;

    // 내 위치 빨간 원
    const myEl = document.createElement('div');
    myEl.style.cssText = `
      width: 20px; height: 20px;
      background: white; border: 3px solid #EF4444;
      border-radius: 50%; display: flex; align-items: center; justify-content: center;
    `;
    const inner = document.createElement('div');
    inner.style.cssText = 'width:8px;height:8px;background:#EF4444;border-radius:50%;';
    myEl.appendChild(inner);
    const myMarker = new window.kakao.maps.CustomOverlay({
      position: origin, content: myEl, zIndex: 5,
    });
    myMarker.setMap(kakaoMap);
    myMarkerRef.current = myMarker;

    // 목적지 마커
    const destEl = document.createElement('div');
    destEl.style.cssText = `
      width: 36px; height: 36px; background: #3B82F6;
      border: 3px solid white; border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-size: 18px; box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    `;
    destEl.innerHTML = '🏠';
    const destMarker = new window.kakao.maps.CustomOverlay({
      position: destPos, content: destEl, zIndex: 5,
    });
    destMarker.setMap(kakaoMap);
    destMarkerRef.current = destMarker;

    kakaoMap.setBounds(new window.kakao.maps.LatLngBounds(origin, destPos));
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

        const canvas = document.createElement('canvas');
        const node   = mapRef.current;
        canvas.width  = node.offsetWidth;
        canvas.height = node.offsetHeight;
        canvas.style.cssText = 'position:absolute;top:0;left:0;pointer-events:none;z-index:3;';
        node.appendChild(canvas);
        canvasRef.current = canvas;

        const redraw = () => drawCanvas();
        const resize = () => {
          canvas.width  = node.offsetWidth;
          canvas.height = node.offsetHeight;
          drawCanvas();
        };
        window.kakao.maps.event.addListener(kakaoMap, 'center_changed', redraw);
        window.kakao.maps.event.addListener(kakaoMap, 'zoom_changed', resize);
        window.kakao.maps.event.addListener(kakaoMap, 'drag', redraw);

        fetch('/seoul_grid.geojson')
          .then(r => r.json())
          .then(g => {
            featuresRef.current = g.features;
            drawCanvas();
            drawRoute(kakaoMap);
          });
      }
    }, 100);
    return () => clearInterval(wait);
  }, []);

  useEffect(() => {
    if (kakaoMapRef.current) drawRoute(kakaoMapRef.current);
  }, [routeMode]);

  const dest = shelter ?? { name: '정신여자고등학교' };
  const info = routeInfo[routeMode];

  return (
    <div style={{ position: 'relative', height: '100%' }}>
      {/* 지도 */}
      <div ref={mapRef} style={{ position: 'absolute', inset: 0 }} />

      {/* 상단 도착지 카드 */}
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
              {dest.name} · {info.distance}km
            </div>
          </div>
        </div>
        <span style={{ fontSize: 20 }}>✏️</span>
      </div>

      {/* 경로 모드 선택 */}
      <div style={{
        position: 'absolute', top: 76, left: 12, right: 12,
        display: 'flex', gap: 8, zIndex: 10,
      }}>
        <button
          onClick={() => setRouteMode('safe')}
          style={{
            flex: 1, padding: '10px', border: 'none', borderRadius: 12,
            cursor: 'pointer', fontSize: 13, fontWeight: 700,
            background: routeMode === 'safe' ? 'white' : 'rgba(255,255,255,0.6)',
            color: routeMode === 'safe' ? '#3B82F6' : '#9CA3AF',
            boxShadow: routeMode === 'safe' ? '0 2px 8px rgba(0,0,0,0.12)' : 'none',
          }}
        >
          🛡️ 안전 우선
          {routeMode === 'safe' && (
            <span style={{
              marginLeft: 6, background: '#DBEAFE', color: '#1D4ED8',
              fontSize: 10, padding: '2px 6px', borderRadius: 10,
            }}>추천</span>
          )}
        </button>
        <button
          onClick={() => setRouteMode('fast')}
          style={{
            flex: 1, padding: '10px', border: 'none', borderRadius: 12,
            cursor: 'pointer', fontSize: 13, fontWeight: 700,
            background: routeMode === 'fast' ? 'white' : 'rgba(255,255,255,0.6)',
            color: routeMode === 'fast' ? '#F59E0B' : '#9CA3AF',
            boxShadow: routeMode === 'fast' ? '0 2px 8px rgba(0,0,0,0.12)' : 'none',
          }}
        >
          ⚡ 최단 시간
        </button>
      </div>

      {/* 하단 경로 정보 */}
      <div style={{
        position: 'absolute', bottom: 0, left: 0, right: 0,
        background: 'white', borderRadius: '20px 20px 0 0',
        boxShadow: '0 -4px 20px rgba(0,0,0,0.1)', zIndex: 10,
        padding: '16px 20px 40px',
      }}>
        <div style={{
          width: 36, height: 4, background: '#E5E7EB',
          borderRadius: 2, margin: '0 auto 16px',
        }} />

        <div style={{ display: 'flex', gap: 14, marginBottom: 16 }}>
          <div style={{
            background: '#F0FDF4', borderRadius: 12, padding: '12px 16px',
            minWidth: 80, textAlign: 'center',
          }}>
            <div style={{ fontSize: 11, color: '#6B7280', marginBottom: 2 }}>예상시간</div>
            <div style={{ fontSize: 24, fontWeight: 800, color: '#111', lineHeight: 1 }}>
              {info.duration}
            </div>
            <div style={{ fontSize: 11, color: '#6B7280' }}>분</div>
            <div style={{ fontSize: 11, color: '#9CA3AF', marginTop: 4 }}>
              {info.distance}km · {routeMode === 'safe' ? '우회' : '직선'}
            </div>
          </div>
          <div style={{ flex: 1, display: 'flex', alignItems: 'center' }}>
            <div>
              <div style={{ fontSize: 14, fontWeight: 700, color: '#111', marginBottom: 4 }}>
                ✅ {info.desc}
              </div>
              <div style={{ fontSize: 12, color: '#6B7280', lineHeight: 1.5 }}>
                {info.detail}
              </div>
            </div>
          </div>
        </div>

        <button style={{
          width: '100%', padding: '15px', background: '#3B82F6', color: 'white',
          border: 'none', borderRadius: 14, fontSize: 16, fontWeight: 700, cursor: 'pointer',
        }}>
          🚶 길안내 시작
        </button>
      </div>
    </div>
  );
};

export default RoutePage;
