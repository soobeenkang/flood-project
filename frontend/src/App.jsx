import { useEffect, useRef, useState } from 'react';

const TIME_OPTIONS = [
  { label: '현재',    value: 'current' },
  { label: '1시간 후', value: '1h' },
  { label: '3시간 후', value: '3h' },
  { label: '6시간 후', value: '6h' },
];

const MOCK_HEATMAP = {
  current: [75, 76, 1510],
  '1h':    [75, 76, 77, 88, 89, 1510, 1511, 1512],
  '3h':    [75, 76, 77, 78, 80, 81, 87, 90, 1508, 1509, 1510, 1511, 1512, 1513],
  '6h':    [75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,
            91, 92, 93, 94, 99, 100,
            1510, 1511, 1512, 1513,
            3201, 3202, 3203, 3204, 3205],
};

// Mock 대피소 데이터 (백엔드 연동 전)
const MOCK_SHELTER = {
  name: '정신여자고등학교',
  distance: 1500,   // m
  duration: 26,     // 분
  lat: 37.5172,
  lng: 127.0473,
};

const getFloodColor = (flood) =>
  flood === 1 ? 'rgba(220, 38, 38, 0.65)' : null;

const fetchHeatmapData = async (t) => {
  await new Promise(r => setTimeout(r, 300));
  const floodedIds = new Set(MOCK_HEATMAP[t] ?? []);
  const map = new Map();
  floodedIds.forEach(id => map.set(id, 1));
  return map;
};

function App() {
  const mapRef      = useRef(null);
  const canvasRef   = useRef(null);
  const kakaoMapRef = useRef(null);
  const featuresRef = useRef(null);
  const floodMapRef = useRef(new Map());
  const rafRef      = useRef(null);
  const polylineRef = useRef(null);
  const markerRef   = useRef(null);
  const myMarkerRef = useRef(null);

  const [selectedTime, setSelectedTime] = useState('current');
  const [showPanel, setShowPanel]       = useState(false);
  const [isLoading, setIsLoading]       = useState(false);
  const [userLocation, setUserLocation] = useState(null);
  const [shelter, setShelter]           = useState(null);

  // ── Canvas 렌더링 ─────────────────────────────────────────────────────
  const drawCanvas = () => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      const canvas   = canvasRef.current;
      const kakaoMap = kakaoMapRef.current;
      const features = featuresRef.current;
      if (!canvas || !kakaoMap || !features) return;

      const ctx    = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const bounds = kakaoMap.getBounds();
      const sw     = bounds.getSouthWest();
      const ne     = bounds.getNorthEast();
      const proj   = kakaoMap.getProjection();

      features.forEach((feature) => {
        const { grid_id, lon, lat } = feature.properties;
        if (lon < sw.getLng() || lon > ne.getLng() ||
            lat < sw.getLat() || lat > ne.getLat()) return;

        const flood = floodMapRef.current.get(grid_id);
        const color = getFloodColor(flood);
        if (!color) return;

        const coords = feature.geometry.coordinates[0];
        ctx.beginPath();
        coords.forEach(([lng, la], i) => {
          const pt = proj.pointFromCoords(new window.kakao.maps.LatLng(la, lng));
          if (i === 0) ctx.moveTo(pt.x, pt.y);
          else         ctx.lineTo(pt.x, pt.y);
        });
        ctx.closePath();
        ctx.fillStyle = color;
        ctx.fill();
      });
    });
  };

  // ── 카카오맵 초기화 ───────────────────────────────────────────────────
  useEffect(() => {
    const waitForKakao = setInterval(() => {
      if (window.kakao && window.kakao.maps) {
        clearInterval(waitForKakao);

        const kakaoMap = new window.kakao.maps.Map(mapRef.current, {
          center: new window.kakao.maps.LatLng(37.557, 126.774),
          level: 7,
        });
        kakaoMapRef.current = kakaoMap;

        const canvas = document.createElement('canvas');
        const node   = mapRef.current;
        canvas.width  = node.offsetWidth;
        canvas.height = node.offsetHeight;
        canvas.style.cssText =
          'position:absolute;top:0;left:0;pointer-events:none;z-index:3;';
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
        window.addEventListener('resize', resize);

        fetch('/seoul_grid.geojson')
          .then(r => r.json())
          .then(async (geojson) => {
            featuresRef.current = geojson.features;
            const map = await fetchHeatmapData('current');
            floodMapRef.current = map;
            drawCanvas();
          })
          .catch(e => console.error('[GeoJSON]', e));

        // 현재 위치 가져오기
        if (navigator.geolocation) {
          navigator.geolocation.getCurrentPosition(
            (pos) => {
              const loc = { lat: pos.coords.latitude, lng: pos.coords.longitude };
              setUserLocation(loc);
            },
            () => {
              // 위치 권한 거부 시 서울 시청 기본값
              setUserLocation({ lat: 37.5665, lng: 126.978 });
            }
          );
        }
      }
    }, 100);

    return () => clearInterval(waitForKakao);
  }, []);

  // ── 시간 슬라이더 변경 ────────────────────────────────────────────────
  useEffect(() => {
    if (!kakaoMapRef.current || !featuresRef.current) return;
    setIsLoading(true);
    fetchHeatmapData(selectedTime)
      .then(map => {
        floodMapRef.current = map;
        drawCanvas();
      })
      .catch(e => console.error('[히트맵]', e))
      .finally(() => setIsLoading(false));
  }, [selectedTime]);

  // ── 대피경로 표시 ─────────────────────────────────────────────────────
  const handleEvacuation = () => {
    const kakaoMap = kakaoMapRef.current;
    if (!kakaoMap || !userLocation) return;

    // 기존 경로/마커 제거
    if (polylineRef.current) polylineRef.current.setMap(null);
    if (markerRef.current)   markerRef.current.setMap(null);
    if (myMarkerRef.current) myMarkerRef.current.setMap(null);

    // TODO: 실제 API 연결 시 교체
    // const res = await fetch('http://localhost:3000/api/v1/evacuation/route', {
    //   method: 'POST', body: JSON.stringify({ origin: userLocation })
    // });
    // const data = await res.json();

    const shelterData = MOCK_SHELTER;
    setShelter(shelterData);

    const origin = new window.kakao.maps.LatLng(userLocation.lat, userLocation.lng);
    const dest   = new window.kakao.maps.LatLng(shelterData.lat, shelterData.lng);

    // 파란 점선 경로
    const polyline = new window.kakao.maps.Polyline({
      path: [origin, dest],
      strokeWeight: 4,
      strokeColor: '#3B82F6',
      strokeOpacity: 1,
      strokeStyle: 'shortdot',
    });
    polyline.setMap(kakaoMap);
    polylineRef.current = polyline;

    // 내 위치 파란 원
    const myOverlay = document.createElement('div');
    myOverlay.style.cssText = `
      width: 16px; height: 16px;
      background: #3B82F6;
      border: 3px solid white;
      border-radius: 50%;
      box-shadow: 0 0 8px rgba(59,130,246,0.6);
    `;
    const myMarker = new window.kakao.maps.CustomOverlay({
      position: origin,
      content: myOverlay,
      zIndex: 5,
    });
    myMarker.setMap(kakaoMap);
    myMarkerRef.current = myMarker;

    // 목적지 집 아이콘
    const destOverlay = document.createElement('div');
    destOverlay.style.cssText = `
      width: 36px; height: 36px;
      background: #22C55E;
      border: 3px solid white;
      border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-size: 18px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    `;
    destOverlay.innerHTML = '🏠';
    const destMarker = new window.kakao.maps.CustomOverlay({
      position: dest,
      content: destOverlay,
      zIndex: 5,
    });
    destMarker.setMap(kakaoMap);
    markerRef.current = destMarker;

    // 경로가 보이도록 지도 범위 조정
    const bounds = new window.kakao.maps.LatLngBounds(origin, dest);
    kakaoMap.setBounds(bounds);

    setShowPanel(true);
  };

  const handleClosePanel = () => {
    setShowPanel(false);
    if (polylineRef.current) polylineRef.current.setMap(null);
    if (markerRef.current)   markerRef.current.setMap(null);
    if (myMarkerRef.current) myMarkerRef.current.setMap(null);
    setShelter(null);
  };

  return (
    <div style={{
      position: 'relative', width: '100%', height: '100dvh',
      overflow: 'hidden', fontFamily: "'Apple SD Gothic Neo', sans-serif",
    }}>

      {/* 지도 */}
      <div ref={mapRef} style={{ position: 'absolute', inset: 0 }} />

      {/* 시간 슬라이더 */}
      {!showPanel && (
        <div style={{
          position: 'absolute', top: 16, left: '50%', transform: 'translateX(-50%)',
          display: 'flex', background: 'white', borderRadius: 50, padding: 4,
          boxShadow: '0 2px 16px rgba(0,0,0,0.18)', gap: 2, zIndex: 10,
          whiteSpace: 'nowrap',
        }}>
          {TIME_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              onClick={() => setSelectedTime(opt.value)}
              disabled={isLoading}
              style={{
                padding: '10px 20px', border: 'none', borderRadius: 50,
                fontSize: 14, fontWeight: selectedTime === opt.value ? 700 : 500,
                cursor: isLoading ? 'not-allowed' : 'pointer',
                background: selectedTime === opt.value ? '#3B82F6' : 'transparent',
                color: selectedTime === opt.value ? 'white' : '#555',
                boxShadow: selectedTime === opt.value ? '0 2px 8px rgba(59,130,246,0.35)' : 'none',
                opacity: isLoading ? 0.7 : 1,
                transition: 'all 0.2s',
              }}
            >
              {opt.label}
            </button>
          ))}
        </div>
      )}

      {isLoading && (
        <div style={{
          position: 'absolute', top: 72, left: '50%', transform: 'translateX(-50%)',
          background: 'rgba(0,0,0,0.55)', color: 'white', fontSize: 12,
          padding: '6px 14px', borderRadius: 20, zIndex: 10,
        }}>
          불러오는 중…
        </div>
      )}

      {/* 대피 경로 안내 버튼 */}
      {!showPanel && (
        <button
          onClick={handleEvacuation}
          style={{
            position: 'absolute', bottom: 32, left: 16, right: 16,
            padding: '18px 0', background: '#22C55E', color: 'white',
            border: 'none', borderRadius: 16, fontSize: 17, fontWeight: 700,
            cursor: 'pointer', display: 'flex', alignItems: 'center',
            justifyContent: 'center', gap: 8,
            boxShadow: '0 4px 20px rgba(34,197,94,0.4)', zIndex: 10,
          }}
        >
          <span style={{ display: 'inline-block', transform: 'rotate(-45deg)', fontSize: 18 }}>✈</span>
          대피 경로 안내
        </button>
      )}

      {/* 대피경로 패널 */}
      {showPanel && shelter && (
        <div style={{
          position: 'absolute', bottom: 0, left: 0, right: 0,
          background: 'white', borderRadius: '20px 20px 0 0',
          boxShadow: '0 -4px 30px rgba(0,0,0,0.15)', zIndex: 20, overflow: 'hidden',
        }}>
          {/* 초록 헤더 */}
          <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            padding: '18px 20px', background: '#22C55E',
          }}>
            <span style={{ color: 'white', fontSize: 17, fontWeight: 700 }}>대피 경로 안내</span>
            <button onClick={handleClosePanel} style={{
              background: 'rgba(255,255,255,0.28)', border: 'none', color: 'white',
              width: 30, height: 30, borderRadius: '50%', cursor: 'pointer',
              fontSize: 14, display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>✕</button>
          </div>

          {/* 목적지 카드 */}
          <div style={{
            display: 'flex', alignItems: 'center', gap: 14,
            padding: '16px 20px', background: '#F0FDF4',
            borderBottom: '1px solid #E5E7EB',
          }}>
            <div style={{
              width: 40, height: 40, borderRadius: '50%',
              background: '#22C55E', display: 'flex',
              alignItems: 'center', justifyContent: 'center', flexShrink: 0,
            }}>
              <span style={{ fontSize: 20 }}>📍</span>
            </div>
            <div>
              <div style={{ fontSize: 16, fontWeight: 700, color: '#111', marginBottom: 4 }}>
                {shelter.name}
              </div>
              <div style={{ fontSize: 13, color: '#6B7280' }}>
                거리 {(shelter.distance / 1000).toFixed(1)}km &nbsp;·&nbsp; 도보 약 {shelter.duration}분
              </div>
            </div>
          </div>

          {/* 안전 안내 */}
          <div style={{
            display: 'flex', gap: 12, margin: '16px 20px', padding: '14px 16px',
            background: '#FFFBEB', borderRadius: 12, border: '1px solid #FDE68A',
          }}>
            <span style={{ fontSize: 18, color: '#F59E0B', flexShrink: 0, marginTop: 1 }}>ⓘ</span>
            <div>
              <div style={{ fontSize: 14, fontWeight: 700, color: '#92400E', marginBottom: 4 }}>안전 안내</div>
              <div style={{ fontSize: 13, color: '#78350F', lineHeight: 1.5 }}>
                침수 위험 지역을 피해 안전한 경로로 이동하세요.
              </div>
            </div>
          </div>

          {/* 범례 */}
          <div style={{ padding: '0 20px 16px', display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 14, color: '#374151' }}>
              <span style={{
                width: 12, height: 12, borderRadius: '50%',
                background: '#3B82F6', flexShrink: 0, display: 'inline-block',
              }} />
              파란색 점선을 따라 이동
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 14, color: '#374151' }}>
              <span style={{ fontSize: 16 }}>🏠</span>
              목적지: {shelter.name}
            </div>
          </div>

          {/* 지도로 돌아가기 */}
          <button onClick={handleClosePanel} style={{
            width: 'calc(100% - 40px)', margin: '0 20px 28px',
            padding: '14px 0', background: '#F3F4F6', border: 'none',
            borderRadius: 12, fontSize: 15, fontWeight: 600, color: '#374151', cursor: 'pointer',
          }}>
            지도로 돌아가기
          </button>
        </div>
      )}
    </div>
  );
}

export default App;