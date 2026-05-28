const HomePage = ({ weather, onNavigate }) => {
  return (
    <div style={{ height: '100%', overflowY: 'auto', background: '#F0F4FF' }}>
      {/* 상단 날씨 헤더 */}
      <div style={{
        background: 'linear-gradient(160deg, #2563EB 0%, #1D4ED8 100%)',
        padding: '20px 20px 32px',
        borderRadius: '0 0 28px 28px',
      }}>
        {/* 안심예보 뱃지 */}
        <div style={{
          display: 'inline-flex', alignItems: 'center', gap: 6,
          background: 'rgba(255,255,255,0.2)', borderRadius: 20,
          padding: '4px 12px', marginBottom: 16,
        }}>
          <span style={{ fontSize: 14 }}>🌊</span>
          <span style={{ color: 'white', fontSize: 13, fontWeight: 600 }}>안심예보</span>
        </div>

        {/* 위치 + 시간 */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 }}>
          <span style={{ fontSize: 14 }}>📍</span>
          <span style={{ color: 'rgba(255,255,255,0.85)', fontSize: 13 }}>
            {weather.location} · {new Date().toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })}
          </span>
        </div>

        {/* 메인 날씨 텍스트 */}
        <div style={{ color: 'white', fontSize: 26, fontWeight: 800, lineHeight: 1.3, marginBottom: 8 }}>
          {weather.description}
        </div>
        <div style={{ color: 'rgba(255,255,255,0.8)', fontSize: 13, lineHeight: 1.6 }}>
          {weather.forecast}
        </div>

        {/* 날씨 지표 */}
        <div style={{ display: 'flex', gap: 8, marginTop: 20 }}>
          {[
            { icon: '🌧️', label: '강수', value: `${weather.rainfall}mm` },
            { icon: '💧', label: '확률', value: `${weather.probability}%` },
            { icon: '🌡️', label: '기온', value: `${weather.temperature}°` },
            { icon: '💨', label: '풍속', value: `${weather.windSpeed}m/s` },
          ].map((item) => (
            <div key={item.label} style={{
              flex: 1, background: 'rgba(255,255,255,0.15)',
              borderRadius: 12, padding: '10px 6px', textAlign: 'center',
            }}>
              <div style={{ fontSize: 16, marginBottom: 2 }}>{item.icon}</div>
              <div style={{ color: 'rgba(255,255,255,0.7)', fontSize: 10, marginBottom: 2 }}>{item.label}</div>
              <div style={{ color: 'white', fontSize: 13, fontWeight: 700 }}>{item.value}</div>
            </div>
          ))}
        </div>
      </div>

      {/* 메뉴 카드 */}
      <div style={{ padding: '20px 16px', display: 'flex', flexDirection: 'column', gap: 12 }}>
        {[
          {
            icon: '🌧️',
            bg: '#EFF6FF',
            title: '침수 예측 지도',
            desc: '우리 동네 침수 위험 한눈에 보기',
            page: 'map',
          },
          {
            icon: '🧭',
            bg: '#F0F9FF',
            title: '침수 피하는 경로',
            desc: '목적지까지 안전한 길 안내',
            page: 'route',
          },
          {
            icon: '🛟',
            bg: '#F0FDF4',
            title: '가까운 대피소 조회',
            desc: '학교 · 공공기관 · 호텔',
            page: 'shelter',
          },
          {
            icon: '🔔',
            bg: '#FFF7ED',
            title: '경보',
            desc: '지금 우리 지역 활성 경보 2건',
            page: 'alert',
            badge: 2,
          },
        ].map((item) => (
          <button
            key={item.page}
            onClick={() => onNavigate(item.page)}
            style={{
              display: 'flex', alignItems: 'center', gap: 14,
              background: 'white', border: 'none', borderRadius: 16,
              padding: '16px', cursor: 'pointer', textAlign: 'left',
              boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
              width: '100%',
            }}
          >
            <div style={{
              width: 52, height: 52, borderRadius: 14,
              background: item.bg,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 26, flexShrink: 0, position: 'relative',
            }}>
              {item.icon}
              {item.badge && (
                <div style={{
                  position: 'absolute', top: -4, right: -4,
                  width: 18, height: 18, background: '#EF4444',
                  borderRadius: '50%', display: 'flex', alignItems: 'center',
                  justifyContent: 'center', fontSize: 10, color: 'white', fontWeight: 700,
                }}>{item.badge}</div>
              )}
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 15, fontWeight: 700, color: '#111', marginBottom: 3 }}>{item.title}</div>
              <div style={{ fontSize: 13, color: '#6B7280' }}>{item.desc}</div>
            </div>
            <span style={{ color: '#D1D5DB', fontSize: 18 }}>›</span>
          </button>
        ))}
      </div>
    </div>
  );
};

export default HomePage;
