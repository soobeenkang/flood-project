import { useEffect, useRef, useState } from 'react';
import { HEATMAP_COLORS, SHELTER_TYPES } from '../data/mockData';
import { getHeatmapGrids, getShelters } from '../services/api';

const ShelterPage = ({ userLocation, onNavigateRoute }) => {
  const mapRef      = useRef(null);
  const canvasRef   = useRef(null);
  const kakaoMapRef = useRef(null);
  const featuresRef = useRef(null);
  const floodIdsRef = useRef(new Set());
  const rafRef      = useRef(null);
  const markersRef  = useRef([]);

  const [filterType, setFilterType]           = useState('all');
  const [selectedShelter, setSelectedShelter] = useState(null);
  const [shelters, setShelters]               = useState([]);
  const [isLoading, setIsLoading]             = useState(false);

  // ── 대피소 API fetch ──────────────────────────────────────────────────
  const fetchShelters = async (type = 'all') => {
    setIsLoading(true);
    try {
      const data = await getShelters(userLocation.lat, userLocation.lng, type, 3000);
      setShelters(data.shelters ?? []);
    } catch (e) {
      console.error('[ShelterPage] 대피소 fetch 실패:', e);
      setShelters([]);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchCurrentHeatmap = async () => {
    try {
      const data = await getHeatmapGrids(userLocation.lat, userLocation.lng, 'now', 5000);
      floodIdsRef.current = new Set(
        (data.grids ?? []).filter(g => g.isFlooded).map(g => g.grid_id)
      );
    } catch (e) {
      console.error('[ShelterPage] 히트맵 fetch 실패:', e);
      floodIdsRef.current = new Set();
    }
  };

  // ── Canvas 렌더링 ─────────────────────────────────────────────────────
  const drawCanvas = () => {
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

  // ── 대피소 마커 ───────────────────────────────────────────────────────
  const addShelterMarkers = (kakaoMap) => {
    markersRef.current.forEach(m => m.setMap(null));
    markersRef.current = [];

    shelters.forEach((shelter) => {
      const isSelected = selectedShelter?.id === shelter.id;
      const el = document.createElement('div');
      el.style.cssText = `
        width: ${isSelected ? 52 : 40}px;
        height: ${isSelected ? 52 : 40}px;
        background: ${isSelected ? '#3B82F6' : 'white'};
        border: 3px solid ${isSelected ? '#2563EB' : '#E5E7EB'};
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: ${isSelected ? 24 : 20}px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        cursor: pointer;
        transition: all 0.2s;
      `;
      el.innerHTML = SHELTER_TYPES[shelter.type]?.emoji ?? '🏢';
      el.addEventListener('click', () => setSelectedShelter(shelter));

      const marker = new window.kakao.maps.CustomOverlay({
        position: new window.kakao.maps.LatLng(
          shelter.lat,
          shelter.lon ?? shelter.lng  // lon 또는 lng 둘 다 대응
        ),
        content: el,
        zIndex: 5,
      });
      marker.setMap(kakaoMap);
      markersRef.current.push(marker);
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

        window.kakao.maps.event.addListener(kakaoMap, 'center_changed', drawCanvas);
        window.kakao.maps.event.addListener(kakaoMap, 'zoom_changed',   drawCanvas);
        window.kakao.maps.event.addListener(kakaoMap, 'dragend',        drawCanvas);
        window.kakao.maps.event.addListener(kakaoMap, 'tilesloaded',    drawCanvas);

        Promise.all([
          fetch('/seoul_grid.geojson').then(r => r.json()),
          fetchCurrentHeatmap(),
          fetchShelters('all'),
        ]).then(([geojson]) => {
          featuresRef.current = geojson.features;
          drawCanvas();
          if (kakaoMapRef.current) addShelterMarkers(kakaoMapRef.current);
        }).catch(e => console.error('[ShelterPage init]', e));
      }
    }, 100);
    return () => clearInterval(wait);
  }, []);

  // ── 필터 변경 시 ──────────────────────────────────────────────────────
  useEffect(() => {
    fetchShelters(filterType).then(() => {
      if (kakaoMapRef.current) addShelterMarkers(kakaoMapRef.current);
    });
  }, [filterType]);

  // ── 선택 변경 시 마커 갱신 ────────────────────────────────────────────
  useEffect(() => {
    if (kakaoMapRef.current) addShelterMarkers(kakaoMapRef.current);
  }, [selectedShelter, shelters]);

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

      {/* 필터 탭 */}
      <div style={{
        position: 'absolute', top: 12, left: 12, right: 12,
        display: 'flex', gap: 8, zIndex: 10,
      }}>
        {Object.entries(SHELTER_TYPES).map(([key, val]) => (
          <button key={key} onClick={() => setFilterType(key)} style={{
            padding: '8px 14px', border: 'none', borderRadius: 20,
            fontSize: 13, fontWeight: 600, cursor: 'pointer',
            background: filterType === key ? '#3B82F6' : 'white',
            color: filterType === key ? 'white' : '#374151',
            boxShadow: '0 1px 4px rgba(0,0,0,0.1)',
            whiteSpace: 'nowrap',
          }}>
            {val.emoji} {val.label}
          </button>
        ))}
      </div>

      {/* 로딩 */}
      {isLoading && (
        <div style={{
          position: 'absolute', top: '50%', left: '50%',
          transform: 'translate(-50%,-50%)',
          background: 'rgba(0,0,0,0.55)', color: 'white',
          padding: '8px 16px', borderRadius: 20, fontSize: 13, zIndex: 10,
        }}>불러오는 중…</div>
      )}

      {/* 선택된 대피소 패널 */}
      {selectedShelter && (
        <div style={{
          position: 'absolute', bottom: 0, left: 0, right: 0,
          background: 'white', borderRadius: '20px 20px 0 0',
          boxShadow: '0 -4px 20px rgba(0,0,0,0.12)', zIndex: 20,
          padding: '16px 20px 40px',
        }}>
          <div style={{
            width: 36, height: 4, background: '#E5E7EB',
            borderRadius: 2, margin: '0 auto 16px',
          }} />
          <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 16 }}>
            <div style={{
              width: 52, height: 52, borderRadius: 14,
              background: '#F3F4F6', display: 'flex',
              alignItems: 'center', justifyContent: 'center', fontSize: 28,
            }}>
              {SHELTER_TYPES[selectedShelter.type]?.emoji ?? '🏢'}
            </div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                <span style={{ fontSize: 17, fontWeight: 700, color: '#111' }}>{selectedShelter.name}</span>
                <span style={{
                  background: '#DCFCE7', color: '#16A34A',
                  fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 20,
                }}>{selectedShelter.status}</span>
              </div>
              <div style={{ fontSize: 13, color: '#6B7280' }}>
                📍 {selectedShelter.distance}m &nbsp;·&nbsp;
                🚶 도보 약 {selectedShelter.walkMinutes}분
              </div>
            </div>
          </div>
          <button onClick={() => onNavigateRoute(selectedShelter)} style={{
            width: '100%', padding: '15px', background: '#3B82F6', color: 'white',
            border: 'none', borderRadius: 14, fontSize: 16, fontWeight: 700, cursor: 'pointer',
          }}>
            🧭 대피 경로 안내
          </button>
        </div>
      )}
    </div>
  );
};

export default ShelterPage;
