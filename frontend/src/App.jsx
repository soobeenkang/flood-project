import { useEffect, useRef, useState } from 'react';

const TIME_OPTIONS = [
  { label: '현재', value: '' },
  { label: '1시간 후', value: '1h' },
  { label: '3시간 후', value: '3h' },
];

const getFloodColor = (flood) => {
  return flood === 1 ? 'rgba(220, 38, 38, 0.65)' : null;
};

// Mock 데이터: grid_id 0~19 중 짝수를 침수(1)로
const MOCK_FLOOD_DATA = Array.from({ length: 10 }, (_, i) => ({
  grid_id: 25091 + i,
  flood: 1,
}));

function App() {
  const mapRef      = useRef(null);
  const canvasRef   = useRef(null);
  const kakaoMapRef = useRef(null);
  const featuresRef = useRef(null);
  const floodMapRef = useRef(new Map());
  const rafRef      = useRef(null);

  const [selectedTime, setSelectedTime] = useState('');
  const [showPanel, setShowPanel]       = useState(false);

  // ── Canvas 렌더링 ──────────────────────────────────────────────────────
  const drawCanvas = () => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      const canvas   = canvasRef.current;
      const kakaoMap = kakaoMapRef.current;
      const features = featuresRef.current;

      if (!canvas || !kakaoMap || !features) {
        console.log('[drawCanvas] 준비 안됨:', { canvas: !!canvas, kakaoMap: !!kakaoMap, features: !!features });
        return;
      }

      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const bounds = kakaoMap.getBounds();
      const sw     = bounds.getSouthWest();
      const ne     = bounds.getNorthEast();
      const proj   = kakaoMap.getProjection();

      let drawn = 0;

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
        drawn++;
      });

      console.log('[drawCanvas] 그린 격자 수:', drawn);
    });
  };

  // ── 카카오맵 + GeoJSON + Mock 데이터 순서 보장 ────────────────────────
  useEffect(() => {
    const waitForKakao = setInterval(() => {
      if (window.kakao && window.kakao.maps) {
        clearInterval(waitForKakao);

        // 1. 카카오맵 초기화
        const kakaoMap = new window.kakao.maps.Map(mapRef.current, {
          center: new window.kakao.maps.LatLng(37.5665, 126.978),
          level: 7,
        });
        kakaoMapRef.current = kakaoMap;

        // 2. Canvas 생성
        const canvas = document.createElement('canvas');
        const node   = mapRef.current;
        canvas.width  = node.offsetWidth;
        canvas.height = node.offsetHeight;
        canvas.style.cssText =
          'position:absolute;top:0;left:0;pointer-events:none;z-index:3;';
        node.appendChild(canvas);
        canvasRef.current = canvas;

        // 3. 지도 이동/줌 이벤트
        const redraw = () => {
          canvas.width  = node.offsetWidth;
          canvas.height = node.offsetHeight;
          drawCanvas();
        };
        window.kakao.maps.event.addListener(kakaoMap, 'center_changed', redraw);
        window.kakao.maps.event.addListener(kakaoMap, 'zoom_changed', redraw);
        window.addEventListener('resize', redraw);

        // 4. Mock floodMap 설정
        const map = new Map();
        MOCK_FLOOD_DATA.forEach(({ grid_id, flood }) => map.set(grid_id, flood));
        floodMapRef.current = map;
        console.log('[Mock] floodMap 설정 완료, 크기:', map.size);

        // 5. GeoJSON fetch → 완료 후 drawCanvas
        fetch('/seoul_grid.geojson')
          .then((r) => r.json())
          .then((geojson) => {
            featuresRef.current = geojson.features;
            console.log('[GeoJSON] 로드 완료, features:', geojson.features.length);
            drawCanvas();
          })
          .catch((e) => console.error('[GeoJSON fetch 실패]', e));
      }
    }, 100);

    return () => clearInterval(waitForKakao);
  }, []);

  // ── 시간 슬라이더 변경 ────────────────────────────────────────────────
  useEffect(() => {
    if (!kakaoMapRef.current || !featuresRef.current) return;
    // TODO: 실제 API 연결 시 여기서 fetch
    drawCanvas();
  }, [selectedTime]);

  return (
    <div style={{
      position: 'relative', width: '100%', height: '100dvh',
      overflow: 'hidden', fontFamily: "'Apple SD Gothic Neo', sans-serif",
    }}>

      {/* 지도 */}
      <div ref={mapRef} style={{ position: 'absolute', inset: 0 }} />

      {/* 시간 슬라이더 */}
      <div style={{
        position: 'absolute', top: 16, left: '50%', transform: 'translateX(-50%)',
        display: 'flex', background: 'white', borderRadius: 50, padding: 4,
        boxShadow: '0 2px 16px rgba(0,0,0,0.18)', gap: 2, zIndex: 10,
      }}>
        {TIME_OPTIONS.map((opt) => (
          <button
            key={opt.value}
            onClick={() => setSelectedTime(opt.value)}
            style={{
              padding: '10px 32px', border: 'none', borderRadius: 50,
              fontSize: 15, fontWeight: selectedTime === opt.value ? 700 : 500,
              cursor: 'pointer', whiteSpace: 'nowrap',
              background: selectedTime === opt.value ? '#3B82F6' : 'transparent',
              color: selectedTime === opt.value ? 'white' : '#555',
              boxShadow: selectedTime === opt.value ? '0 2px 8px rgba(59,130,246,0.35)' : 'none',
            }}
          >
            {opt.label}
          </button>
        ))}
      </div>

      {/* 대피 경로 안내 버튼 */}
      {!showPanel && (
        <button
          onClick={() => setShowPanel(true)}
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
      {showPanel && (
        <div style={{
          position: 'absolute', bottom: 0, left: 0, right: 0,
          background: 'white', borderRadius: '20px 20px 0 0',
          boxShadow: '0 -4px 30px rgba(0,0,0,0.15)', zIndex: 20, overflow: 'hidden',
        }}>
          <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            padding: '18px 20px 14px', background: '#22C55E',
          }}>
            <span style={{ color: 'white', fontSize: 17, fontWeight: 700 }}>대피 경로 안내</span>
            <button onClick={() => setShowPanel(false)} style={{
              background: 'rgba(255,255,255,0.28)', border: 'none', color: 'white',
              width: 28, height: 28, borderRadius: '50%', cursor: 'pointer', fontSize: 14,
            }}>✕</button>
          </div>

          <div style={{
            display: 'flex', alignItems: 'center', gap: 14,
            padding: '16px 20px', background: '#F0FDF4', borderBottom: '1px solid #E5E7EB',
          }}>
            <span style={{ fontSize: 28 }}>📍</span>
            <div>
              <div style={{ fontSize: 16, fontWeight: 700, color: '#111', marginBottom: 4 }}>정신여자고등학교</div>
              <div style={{ fontSize: 13, color: '#6B7280' }}>거리 1.5km &nbsp;·&nbsp; 도보 약 26분</div>
            </div>
          </div>

          <div style={{
            display: 'flex', gap: 12, margin: '16px 20px', padding: '14px 16px',
            background: '#FFFBEB', borderRadius: 12, border: '1px solid #FDE68A',
          }}>
            <span style={{ fontSize: 18, color: '#F59E0B', flexShrink: 0 }}>ⓘ</span>
            <div>
              <div style={{ fontSize: 14, fontWeight: 700, color: '#92400E', marginBottom: 4 }}>안전 안내</div>
              <div style={{ fontSize: 13, color: '#78350F', lineHeight: 1.5 }}>
                침수 위험 지역을 피해 안전한 경로로 이동하세요.
              </div>
            </div>
          </div>

          <div style={{ padding: '0 20px 16px', display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 14, color: '#374151' }}>
              <span style={{ width: 12, height: 12, borderRadius: '50%', background: '#3B82F6', flexShrink: 0, display: 'inline-block' }} />
              파란색 점선을 따라 이동
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 14, color: '#374151' }}>
              <span style={{ fontSize: 16 }}>🏠</span>
              목적지: 정신여자고등학교
            </div>
          </div>

          <button onClick={() => setShowPanel(false)} style={{
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
