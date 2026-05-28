import { useState } from 'react';
import FloodAlert from './components/FloodAlert';
import HomePage from './pages/HomePage';
import MapPage from './pages/MapPage';
import ShelterPage from './pages/ShelterPage';
import RoutePage from './pages/RoutePage';
import GuidelinesPage from './pages/GuidelinesPage';
import { MOCK_WEATHER, getRandomUserLocation } from './data/mockData';

const NAV_ITEMS = [
  { id: 'home',       emoji: '🏠', label: '홈' },
  { id: 'map',        emoji: '🌧️', label: '침수지도' },
  { id: 'shelter',    emoji: '🛟', label: '대피소' },
  { id: 'route',      emoji: '🧭', label: '경로' },
  { id: 'guidelines', emoji: '📖', label: '지침' },
  { id: 'alert',      emoji: '🔔', label: '경보', badge: 2 },
];

const USER_LOCATION = getRandomUserLocation();

function App() {
  const [page, setPage]                       = useState('home');
  const [showAlert, setShowAlert]             = useState(MOCK_WEATHER.isFlooded);
  const [selectedShelter, setSelectedShelter] = useState(null);

  const handleShelterToRoute = (shelter) => {
    setSelectedShelter(shelter);
    setPage('route');
  };

  const renderPage = () => {
    switch (page) {
      case 'home':       return <HomePage weather={MOCK_WEATHER} onNavigate={setPage} />;
      case 'map':        return <MapPage userLocation={USER_LOCATION} />;
      case 'shelter':    return <ShelterPage userLocation={USER_LOCATION} onNavigateRoute={handleShelterToRoute} />;
      case 'route':      return <RoutePage userLocation={USER_LOCATION} shelter={selectedShelter} />;
      case 'guidelines': return <GuidelinesPage />;
      default:           return <HomePage weather={MOCK_WEATHER} onNavigate={setPage} />;
    }
  };

  return (
    <div style={{
      width: '100%', height: '100dvh',
      display: 'flex', fontFamily: "'Apple SD Gothic Neo', 'Pretendard', sans-serif",
      background: '#F8FAFC', overflow: 'hidden',
    }}>
      {showAlert && (
        <FloodAlert
          weather={MOCK_WEATHER}
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
            <span style={{ fontSize: 10, fontWeight: 600, color: page === item.id ? '#3B82F6' : '#9CA3AF' }}>
              {item.label}
            </span>
            {item.badge && (
              <div style={{
                position: 'absolute', top: 6, right: 6,
                width: 16, height: 16, background: '#EF4444',
                borderRadius: '50%', display: 'flex', alignItems: 'center',
                justifyContent: 'center', fontSize: 9, color: 'white', fontWeight: 700,
              }}>{item.badge}</div>
            )}
          </button>
        ))}

        <div style={{ flex: 1 }} />
        <button style={{
          width: 56, padding: '10px 0', border: 'none', borderRadius: 14,
          cursor: 'pointer', background: 'transparent',
          display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4, marginBottom: 16,
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
