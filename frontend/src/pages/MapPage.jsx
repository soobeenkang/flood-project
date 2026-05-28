import { useEffect, useRef, useState } from 'react';
import { MOCK_HEATMAP, HEATMAP_COLORS } from '../data/mockData';

const TIME_STEPS = [
  { label: '현재',  value: 'current' },
  { label: '1시간', value: '1h' },
  { label: '3시간', value: '3h' },
  { label: '6시간', value: '6h' },
];

const SHOW_FROM = {
  current: ['current'],
  '1h':    ['1h', 'current'],
  '3h':    ['3h', '1h', 'current'],
  '6h':    ['6h', '3h', '1h', 'current'],
};

const LAYER_ORDER = ['6h', '3h', '1h', 'current'];

const MapPage = ({ userLocation }) => {
  const mapRef          = useRef(null);
  const canvasRef       = useRef(null);
  const kakaoMapRef     = useRef(null);
  const featuresRef     = useRef(null);
  const rafRef          = useRef(null);
  const selectedTimeRef = useRef('current');

  const [selectedTime, setSelectedTime] = useState('current');
  const [isLoading, setIsLoading]       = useState(false);

  // ── Canvas 렌더링 ──────────────────────────────────────────────────────
  const redraw = () => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      const canvas   = canvasRef.current;
      const kakaoMap = kakaoMapRef.current;
      const features = featuresRef.current;
      if (!canvas || !kakaoMap || !features) return;

      // canvas 크기를 지도 컨테이너에 맞춤
      const node = mapRef.current;
      canvas.width  = node.offsetWidth;
      canvas.height = node.offsetHeight;

      const ctx    = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const bounds = kakaoMap.getBounds();
      const sw     = bounds.getSouthWest();
      const ne     = bounds.getNorthEast();
      const proj   = kakaoMap.getProjection();

      const t            = selectedTimeRef.current;
      const layersToShow = new Set(SHOW_FROM[t] ?? ['current']);

      LAYER_ORDER.forEach((layer) => {
        if (!layersToShow.has(layer)) return;
        const ids   = new Set(MOCK_HEATMAP[layer] ?? []);
        const color = HEATMAP_COLORS[layer];

        features.forEach((feature) => {
          const { grid_id, lon, lat } = feature.properties;
          if (!ids.has(grid_id)) return;
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
          ctx.fillStyle = color;
          ctx.fill();
        });
      });
    });
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

        // Canvas 생성 - mapRef 안에 붙임
        const canvas = document.createElement('canvas');
        const node   = mapRef.current;
        node.style.position = 'relative';
        canvas.width  = node.offsetWidth;
        canvas.height = node.offsetHeight;
        canvas.style.cssText =
          'position:absolute;top:0;left:0;pointer-events:none;z-index:3;';
        node.appendChild(canvas);
        canvasRef.current = canvas;

        // dragend: 드래그 끝난 후 재렌더 (좌표 정확)
        window.kakao.maps.event.addListener(kakaoMap, 'dragend', redraw);
        window.kakao.maps.event.addListener(kakaoMap, 'zoom_changed', redraw);
        window.kakao.maps.event.addListener(kakaoMap, 'tilesloaded', redraw);
        window.addEventListener('resize', redraw);

        // GeoJSON 로드 후 초기 렌더
        fetch('/seoul_grid.geojson')
          .then(r => r.json())
          .then(g => {
            featuresRef.current = g.features;
            redraw();
          })
          .catch(e => console.error('[GeoJSON]', e));
      }
    }, 100);

    return () => clearInterval(wait);
  }, []);

  // ── 시간 슬라이더 변경 ────────────────────────────────────────────────
  const handleTimeChange = (t) => {
    setSelectedTime(t);
    selectedTimeRef.current = t;
    setIsLoading(true);
    setTimeout(() => {
      redraw();
      setIsLoading(false);
    }, 300);
  };

  // ─────────────────────────────────────────────────────────────────────
  return (
    <div style={{ position: 'relative', height: '100%' }}>

      {/* 지도 */}
      <div ref={mapRef} style={{ position: 'absolute', inset: 0 }} />

      {/* 상단 헤더 카드 */}
      <div style={{
        position: 'absolute', top: 12, left: 12, right: 12,
        background: 'white', borderRadius: 16, padding: '12px 16px',
        boxShadow: '0 2px 12px rgba(0,0,0,0.12)', zIndex: 10,
        display: 'flex', alignItems: 'center', gap: 12,
      }}>
        <div style={{
          width: 40, height: 40, borderRadius: 12,
          background: '#EFF6FF', display: 'flex',
          alignItems: 'center', justifyContent: 'center', fontSize: 20,
        }}>🌧️</div>
        <div>
          <div style={{ fontSize: 11, color: '#9CA3AF', fontWeight: 600 }}>침수 예측 · 서울</div>
          <div style={{ fontSize: 15, fontWeight: 700, color: '#111' }}>지금 침수된 지역</div>
        </div>
      </div>

      {/* 슬라이더 카드 */}
      <div style={{
        position: 'absolute', top: 80, left: 12, right: 12,
        background: 'white', borderRadius: 16, padding: '14px 16px',
        boxShadow: '0 2px 12px rgba(0,0,0,0.12)', zIndex: 10,
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
          <span style={{ fontSize: 13, color: '#6B7280', fontWeight: 500 }}>예측 시점</span>
          <span style={{ fontSize: 13, color: '#3B82F6', fontWeight: 600 }}>
            {TIME_STEPS.find(t => t.value === selectedTime)?.label}
          </span>
        </div>
        <div style={{ display: 'flex' }}>
          {TIME_STEPS.map((step) => (
            <button
              key={step.value}
              onClick={() => handleTimeChange(step.value)}
              style={{
                flex: 1, padding: '8px 4px', border: 'none',
                borderRadius: 8, cursor: 'pointer', fontSize: 12, fontWeight: 600,
                background: selectedTime === step.value ? '#3B82F6' : 'transparent',
                color: selectedTime === step.value ? 'white' : '#9CA3AF',
                transition: 'all 0.2s',
              }}
            >
              {step.label}
            </button>
          ))}
        </div>
      </div>

      {isLoading && (
        <div style={{
          position: 'absolute', top: '50%', left: '50%',
          transform: 'translate(-50%,-50%)',
          background: 'rgba(0,0,0,0.55)', color: 'white',
          padding: '8px 16px', borderRadius: 20, fontSize: 13, zIndex: 10,
        }}>불러오는 중…</div>
      )}

      {/* 하단 시트 */}
      <div style={{
        position: 'absolute', bottom: 0, left: 0, right: 0,
        background: 'white', borderRadius: '20px 20px 0 0',
        boxShadow: '0 -4px 20px rgba(0,0,0,0.1)', zIndex: 10,
        padding: '16px 20px 32px',
      }}>
        <div style={{
          width: 36, height: 4, background: '#E5E7EB',
          borderRadius: 2, margin: '0 auto 16px',
        }} />
        <div style={{ fontSize: 17, fontWeight: 700, color: '#111', marginBottom: 4 }}>실시간 침수 상황</div>
        <div style={{ fontSize: 13, color: '#6B7280', marginBottom: 14 }}>실시간 감지된 침수 격자</div>
        <div style={{ display: 'flex', gap: 12, marginBottom: 20 }}>
          {TIME_STEPS.map((step) => (
            <div key={step.value} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <div style={{
                width: 14, height: 14, borderRadius: 3,
                background: HEATMAP_COLORS[step.value],
              }} />
              <span style={{ fontSize: 12, color: '#374151', fontWeight: 500 }}>{step.label}</span>
            </div>
          ))}
        </div>
        <div style={{ display: 'flex', gap: 10 }}>
          <button style={{
            flex: 1, padding: '14px', background: '#3B82F6', color: 'white',
            border: 'none', borderRadius: 12, fontSize: 14, fontWeight: 700, cursor: 'pointer',
          }}>
            🛟 가까운 대피소
          </button>
          <button style={{
            flex: 1, padding: '14px', background: '#F3F4F6', color: '#374151',
            border: 'none', borderRadius: 12, fontSize: 14, fontWeight: 700, cursor: 'pointer',
          }}>
            🔔 이 지역 알림
          </button>
        </div>
      </div>
    </div>
  );
};

export default MapPage;
