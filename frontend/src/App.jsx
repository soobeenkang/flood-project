import { useState, useEffect } from 'react';
import FloodAlert from './components/FloodAlert';
import HomePage from './pages/HomePage';
import MapPage from './pages/MapPage';
import ShelterPage from './pages/ShelterPage';
import RoutePage from './pages/RoutePage';
import GuidelinesPage from './pages/GuidelinesPage';
import AlertsPage from './pages/AlertsPage';
import { MOCK_WEATHER, getRandomUserLocation } from './data/mockData';
import { checkFlood, getWeather, getAlerts } from './services/api';

const NAV_ITEMS = [
  { id: 'home',        emoji: '🏠', label: '홈' },
  { id: 'map',         emoji: '🌧️', label: '침수지도' },
  { id: 'shelter',     emoji: '🛟', label: '대피소' },
  { id: 'route',       emoji: '🧭', label: '경로' },
  { id: 'guidelines',  emoji: '📖', label: '지침' },
  { id: 'alert',       emoji: '🔔', label: '경보', badge: true },
];

const USER_LOCATION = getRandomUserLocation();
const USE_MOCK = false; // 백엔드 준비되면 false로 변경

function App() {
  const [page, setPage]                       = useState('home');
  const [showAlert, setShowAlert]             = useState(false);
  const [selectedShelter, setSelectedShelter] = useState(null);
  const [weather, setWeather]                 = useState(MOCK_WEATHER);
  const [alertBadge, setAlertBadge]           = useState(0);

  // ── 앱 초기화: 위치 기반 침수 여부 + 날씨 fetch ──────────────────────
  useEffect(() => {
    const { lat, lng } = USER_LOCATION;

    // 경보 건수
    getAlerts(lat, lng, 50)
      .then(data => setAlertBadge(data.alerts?.length ?? 0))
      .catch(() => setAlertBadge(0));

    if (USE_MOCK) {
      setShowAlert(MOCK_WEATHER.isFlooded);
      return;
    }

    // 침수 여부 확인
    checkFlood(lat, lng)
      .then(data => setShowAlert(data.isFlooded))
      .catch(() => setShowAlert(false));

    // 날씨 정보
    getWeather(lat, lng)
      .then(data => setWeather({
        district:    data.district,
        rainfall:    data.rainfall,
        rainProb:    data.rainProb,
        temperature: data.temperature,
        windSpeed:   data.windSpeed,
        forecast:    data.forecast,
        isFlooded:   false,
        floodDepth:  0,
      }))
      .catch(() => {}); // 실패 시 Mock 유지

  }, []);

  const handleShelterToRoute = (shelter) => {
    setSelectedShelter(shelter);
    setPage('route');
  };

  const renderPage = () => {
    switch (page) {
      case 'home':       return <HomePage weather={weather} alertBadge={alertBadge} onNavigate={setPage} />;
      case 'map':        return <MapPage userLocation={USER_LOCATION} onNavigateShelter={() => setPage('shelter')} />;
      case 'shelter':    return <ShelterPage userLocation={USER_LOCATION} onNavigateRoute={handleShelterToRoute} />;
      case 'route':      return <RoutePage userLocation={USER_LOCATION} shelter={selectedShelter} />;
      case 'guidelines': return <GuidelinesPage />;
      case 'alert':      return <AlertsPage userLocation={USER_LOCATION} />;
      default:           return <HomePage weather={weather} onNavigate={setPage} />;
    }
  };

  return (
    <div style={{
      width: '100%', height: '100dvh',
      display: 'flex', fontFamily: "'Apple SD Gothic Neo', 'Pretendard', sans-serif",
      background: '#F8FAFC', overflow: 'hidden',
    }}>
      {/* 침수 경보 팝업 */}
      {showAlert && (
        <FloodAlert
          weather={weather}
          onClose={() => setShowAlert(false)}
          onFindShelter={() => { setShowAlert(false); setPage('shelter'); }}
          onFindRoute={() => { setShowAlert(false); setPage('route'); }}
        />
      )}

      {/* 좌측 네비게이션 */}
      <div style={{
        width: 72, background: 'white', borderRight: '1px solid #F3F4F6',
        display: 'flex', flexDirection: 'column', alignItems: 'center',
        paddingTop: 20, gap: 4, flexShrink: 0,
        boxShadow: '2px 0 8px rgba(0,0,0,0.04)',
      }}>
        {NAV_ITEMS.map((item) => (
          <button
            key={item.id}
            onClick={() => setPage(item.id)}
            style={{
              width: 56, padding: '10px 0', border: 'none', borderRadius: 14,
              cursor: 'pointer', background: page === item.id ? '#EFF6FF' : 'transparent',
              display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4,
              position: 'relative',
            }}
          >
            <div style={{ fontSize: 22, filter: page === item.id ? 'none' : 'grayscale(50%)' }}>
              {item.emoji}
            </div>
            <span style={{
              fontSize: 10, fontWeight: 600,
              color: page === item.id ? '#3B82F6' : '#9CA3AF',
            }}>{item.label}</span>
            {item.badge && alertBadge > 0 && (
              <div style={{
                position: 'absolute', top: 6, right: 6,
                width: 16, height: 16, background: '#EF4444',
                borderRadius: '50%', display: 'flex', alignItems: 'center',
                justifyContent: 'center', fontSize: 9, color: 'white', fontWeight: 700,
              }}>{alertBadge}</div>
            )}
          </button>
        ))}

        <div style={{ flex: 1 }} />
        <button style={{
          width: 56, padding: '10px 0', border: 'none', borderRadius: 14,
          cursor: 'pointer', background: 'transparent',
          display: 'flex', flexDirection: 'column', alignItems: 'center',
          gap: 4, marginBottom: 16,
        }}>
          <span style={{ fontSize: 22 }}>⚙️</span>
          <span style={{ fontSize: 10, fontWeight: 600, color: '#9CA3AF' }}>설정</span>
        </button>
      </div>

      {/* 메인 콘텐츠 */}
      <div style={{ flex: 1, overflow: 'hidden', position: 'relative' }}>
        {renderPage()}
      </div>
    </div>
  );
}

export default App;
