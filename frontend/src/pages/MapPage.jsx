import { useEffect, useRef, useState } from 'react';
import { MOCK_HEATMAP, HEATMAP_COLORS } from '../data/mockData';
import { getHeatmapGrids, subscribeToGrid } from '../services/api';

const USE_MOCK = false;

const TIME_STEPS = [
  { label: '현재',  value: 'now' },
  { label: '1시간', value: '1h' },
  { label: '3시간', value: '3h' },
  { label: '6시간', value: '6h' },
];

const SHOW_FROM = {
  now:  ['now'],
  '1h': ['1h', 'now'],
  '3h': ['3h', '1h', 'now'],
  '6h': ['6h', '3h', '1h', 'now'],
};
const LAYER_ORDER = ['6h', '3h', '1h', 'now'];

const MapPage = ({ userLocation, onNavigateShelter }) => {
  const mapRef          = useRef(null);
  const canvasRef       = useRef(null);
  const kakaoMapRef     = useRef(null);
  const featuresRef     = useRef(null);
  const floodIdsRef     = useRef({ now: new Set(), '1h': new Set(), '3h': new Set(), '6h': new Set() });
  const rafRef          = useRef(null);
  const selectedTimeRef = useRef('now');
  const clickListenerRef = useRef(null);

  const [selectedTime, setSelectedTime]     = useState('now');
  const [isLoading, setIsLoading]           = useState(false);

  // 이 지역 알림 관련 상태
  const [alertMode, setAlertMode]           = useState(false);  // 그리드 선택 모드
  const [selectedGridId, setSelectedGridId] = useState(null);
  const [email, setEmail]                   = useState('');
  const [emailStep, setEmailStep]           = useState(false);  // 이메일 입력 단계
  const [submitStatus, setSubmitStatus]     = useState(null);   // 'success' | 'error'

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

      const t            = selectedTimeRef.current;
      const layersToShow = new Set(SHOW_FROM[t] ?? ['now']);

      LAYER_ORDER.forEach((layer) => {
        if (!layersToShow.has(layer)) return;
        const ids   = floodIdsRef.current[layer];
        const color = HEATMAP_COLORS[layer];

        features.forEach((feature) => {
          const { grid_id, lon, lat } = feature.properties;
          if (!ids.has(grid_id)) return;
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
          ctx.fillStyle = color;
          ctx.fill();
        });
      });

      // 선택된 그리드 강조
      if (selectedGridId !== null && features) {
        const feature = features.find(f => f.properties.grid_id === selectedGridId);
        if (feature) {
          const coords = feature.geometry.coordinates[0];
          ctx.beginPath();
          coords.forEach(([lng, la], i) => {
            const x = lngToX(lng);
            const y = latToY(la);
            if (i === 0) ctx.moveTo(x, y);
            else         ctx.lineTo(x, y);
          });
          ctx.closePath();
          ctx.fillStyle   = 'rgba(59, 130, 246, 0.5)';
          ctx.strokeStyle = '#3B82F6';
          ctx.lineWidth   = 2;
          ctx.fill();
          ctx.stroke();
        }
      }
    });
  };

  const fetchHeatmap = async (horizon) => {
    if (USE_MOCK) {
      floodIdsRef.current[horizon] = new Set(MOCK_HEATMAP[horizon] ?? []);
      return;
    }
    const { lat, lng } = userLocation;
    const data = await getHeatmapGrids(lat, lng, horizon, 5000);
    floodIdsRef.current[horizon] = new Set(
      data.grids.filter(g => g.isFlooded).map(g => g.grid_id)
    );
  };

  // 지도 클릭 → 그리드 선택
  const handleMapClick = (kakaoMap) => {
    return window.kakao.maps.event.addListener(kakaoMap, 'click', (e) => {
      const features = featuresRef.current;
      if (!features) return;

      const clickLat = e.latLng.getLat();
      const clickLng = e.latLng.getLng();

      // 클릭 위치가 포함된 그리드 찾기
      const found = features.find((feature) => {
        const { lon, lat } = feature.properties;
        // 간단히 중심 좌표 기준 ±0.001 범위로 체크
        return Math.abs(lat - clickLat) < 0.0009 && Math.abs(lon - clickLng) < 0.0009;
      });

      if (found) {
        setSelectedGridId(found.properties.grid_id);
        setEmailStep(true);
        redraw();
      }
    });
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
        window.addEventListener('resize', redraw);

        Promise.all([
          fetch('/seoul_grid.geojson').then(r => r.json()),
          fetchHeatmap('now'),
          fetchHeatmap('1h'),
          fetchHeatmap('3h'),
          fetchHeatmap('6h'),
        ]).then(([geojson]) => {
          featuresRef.current = geojson.features;
          redraw();
        }).catch(e => console.error('[MapPage init]', e));
      }
    }, 100);
    return () => clearInterval(wait);
  }, []);

  // 알림 모드 진입/해제 시 클릭 이벤트 등록/해제
  useEffect(() => {
    const kakaoMap = kakaoMapRef.current;
    if (!kakaoMap) return;

    if (alertMode) {
      clickListenerRef.current = handleMapClick(kakaoMap);
    } else {
      if (clickListenerRef.current) {
        window.kakao.maps.event.removeListener(kakaoMap, 'click', clickListenerRef.current);
        clickListenerRef.current = null;
      }
      setSelectedGridId(null);
      setEmailStep(false);
      setEmail('');
      setSubmitStatus(null);
      redraw();
    }
  }, [alertMode]);

  const handleTimeChange = (t) => {
    setSelectedTime(t);
    selectedTimeRef.current = t;
    setIsLoading(true);
    setTimeout(() => { redraw(); setIsLoading(false); }, 100);
  };

  const handleShelterClick = (event) => {
    event.preventDefault();
    event.stopPropagation();
    onNavigateShelter?.();
  };

  // 구독 등록
  const handleSubscribe = async () => {
    if (!email || !selectedGridId) return;
    try {
      if (USE_MOCK) {
        await new Promise(r => setTimeout(r, 500));
        setSubmitStatus('success');
        return;
      }
      await subscribeToGrid(selectedGridId, email);
      setSubmitStatus('success');
    } catch {
      setSubmitStatus('error');
    }
  };

  return (
    <div style={{ position: 'relative', height: '100%' }}>
      <div style={{ position: 'absolute', inset: 0 }}>
        <div ref={mapRef} style={{ width: '100%', height: '100%', cursor: alertMode ? 'crosshair' : 'default' }} />
        <canvas ref={canvasRef} style={{
          position: 'absolute', top: 0, left: 0,
          width: '100%', height: '100%',
          pointerEvents: 'none', zIndex: 3,
        }} />
      </div>

      {/* 상단 헤더 */}
      {!alertMode && (
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
      )}

      {/* 알림 모드 안내 배너 */}
      {alertMode && !emailStep && (
        <div style={{
          position: 'absolute', top: 12, left: 12, right: 12,
          background: '#3B82F6', borderRadius: 16, padding: '14px 16px',
          boxShadow: '0 2px 12px rgba(59,130,246,0.3)', zIndex: 10,
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        }}>
          <div>
            <div style={{ color: 'white', fontWeight: 700, fontSize: 14, marginBottom: 2 }}>
              알림 받을 지역을 선택해주세요
            </div>
            <div style={{ color: 'rgba(255,255,255,0.8)', fontSize: 12 }}>
              지도에서 원하는 격자를 클릭하세요
            </div>
          </div>
          <button onClick={() => setAlertMode(false)} style={{
            background: 'rgba(255,255,255,0.2)', border: 'none', color: 'white',
            width: 28, height: 28, borderRadius: '50%', cursor: 'pointer', fontSize: 14,
          }}>✕</button>
        </div>
      )}

      {/* 슬라이더 */}
      {!alertMode && (
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
              <button key={step.value} onClick={() => handleTimeChange(step.value)} style={{
                flex: 1, padding: '8px 4px', border: 'none', borderRadius: 8,
                cursor: 'pointer', fontSize: 12, fontWeight: 600,
                background: selectedTime === step.value ? '#3B82F6' : 'transparent',
                color: selectedTime === step.value ? 'white' : '#9CA3AF',
                transition: 'all 0.2s',
              }}>{step.label}</button>
            ))}
          </div>
        </div>
      )}

      {isLoading && (
        <div style={{
          position: 'absolute', top: '50%', left: '50%',
          transform: 'translate(-50%,-50%)',
          background: 'rgba(0,0,0,0.55)', color: 'white',
          padding: '8px 16px', borderRadius: 20, fontSize: 13, zIndex: 10,
        }}>불러오는 중…</div>
      )}

      {/* 이메일 입력 패널 */}
      {alertMode && emailStep && (
        <div style={{
          position: 'absolute', bottom: 0, left: 0, right: 0,
          background: 'white', borderRadius: '20px 20px 0 0',
          boxShadow: '0 -4px 20px rgba(0,0,0,0.12)', zIndex: 20,
          padding: '20px 20px 40px',
        }}>
          <div style={{ width: 36, height: 4, background: '#E5E7EB', borderRadius: 2, margin: '0 auto 16px' }} />

          {submitStatus === 'success' ? (
            <div style={{ textAlign: 'center', padding: '16px 0' }}>
              <div style={{ fontSize: 32, marginBottom: 12 }}>✅</div>
              <div style={{ fontSize: 16, fontWeight: 700, color: '#111', marginBottom: 8 }}>
                알림 등록 완료
              </div>
              <div style={{ fontSize: 13, color: '#6B7280', marginBottom: 20 }}>
                침수 감지 시 이메일로 알려드릴게요
              </div>
              <button onClick={() => setAlertMode(false)} style={{
                width: '100%', padding: '14px', background: '#3B82F6', color: 'white',
                border: 'none', borderRadius: 12, fontSize: 15, fontWeight: 700, cursor: 'pointer',
              }}>확인</button>
            </div>
          ) : (
            <>
              <div style={{ fontSize: 16, fontWeight: 700, color: '#111', marginBottom: 4 }}>
                이메일 알림 등록
              </div>
              <div style={{ fontSize: 13, color: '#6B7280', marginBottom: 16 }}>
                격자 ID: {selectedGridId} · 침수 감지 시 이메일로 알림을 보내드려요
              </div>

              <input
                type="email"
                placeholder="이메일 주소 입력"
                value={email}
                onChange={e => setEmail(e.target.value)}
                style={{
                  width: '100%', padding: '14px 16px',
                  border: '1.5px solid #E5E7EB', borderRadius: 12,
                  fontSize: 15, outline: 'none', marginBottom: 10,
                  boxSizing: 'border-box',
                  borderColor: submitStatus === 'error' ? '#EF4444' : '#E5E7EB',
                }}
              />

              {submitStatus === 'error' && (
                <div style={{ fontSize: 12, color: '#EF4444', marginBottom: 10 }}>
                  등록에 실패했어요. 다시 시도해주세요.
                </div>
              )}

              <div style={{ display: 'flex', gap: 10 }}>
                <button onClick={() => { setEmailStep(false); setSelectedGridId(null); redraw(); }} style={{
                  flex: 1, padding: '14px', background: '#F3F4F6', color: '#374151',
                  border: 'none', borderRadius: 12, fontSize: 14, fontWeight: 600, cursor: 'pointer',
                }}>다시 선택</button>
                <button onClick={handleSubscribe} style={{
                  flex: 2, padding: '14px', background: '#3B82F6', color: 'white',
                  border: 'none', borderRadius: 12, fontSize: 14, fontWeight: 700, cursor: 'pointer',
                  opacity: email ? 1 : 0.5,
                }}>알림 등록</button>
              </div>
            </>
          )}
        </div>
      )}

      {/* 하단 시트 */}
      {!alertMode && (
        <div style={{
          position: 'absolute', bottom: 0, left: 0, right: 0,
          background: 'white', borderRadius: '20px 20px 0 0',
          boxShadow: '0 -4px 20px rgba(0,0,0,0.1)', zIndex: 10,
          padding: '16px 20px 32px',
        }}>
          <div style={{ width: 36, height: 4, background: '#E5E7EB', borderRadius: 2, margin: '0 auto 16px' }} />
          <div style={{ fontSize: 17, fontWeight: 700, color: '#111', marginBottom: 4 }}>실시간 침수 상황</div>
          <div style={{ fontSize: 13, color: '#6B7280', marginBottom: 14 }}>실시간 감지된 침수 격자</div>
          <div style={{ display: 'flex', gap: 12, marginBottom: 20 }}>
            {TIME_STEPS.map((step) => (
              <div key={step.value} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <div style={{ width: 14, height: 14, borderRadius: 3, background: HEATMAP_COLORS[step.value] }} />
                <span style={{ fontSize: 12, color: '#374151', fontWeight: 500 }}>{step.label}</span>
              </div>
            ))}
          </div>
          <div style={{ display: 'flex', gap: 10 }}>
            <button onClick={handleShelterClick} style={{
              flex: 1, padding: '14px', background: '#3B82F6', color: 'white',
              border: 'none', borderRadius: 12, fontSize: 14, fontWeight: 700, cursor: 'pointer',
            }}>🛟 가까운 대피소</button>
            <button
              onClick={() => setAlertMode(true)}
              style={{
                flex: 1, padding: '14px', background: '#F3F4F6', color: '#374151',
                border: 'none', borderRadius: 12, fontSize: 14, fontWeight: 700, cursor: 'pointer',
              }}>🔔 이 지역 알림</button>
          </div>
        </div>
      )}
    </div>
  );
};

export default MapPage;
