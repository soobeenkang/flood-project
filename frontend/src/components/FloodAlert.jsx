const FloodAlert = ({ weather, onClose, onFindShelter, onFindRoute }) => {
  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 100,
      display: 'flex', flexDirection: 'column',
      background: '#fff',
    }}>
      {/* 빨간 경보 헤더 */}
      <div style={{
        background: 'linear-gradient(160deg, #DC2626 0%, #B91C1C 100%)',
        padding: 'calc(env(safe-area-inset-top, 0px) + 76px) 24px 36px',
        display: 'flex', flexDirection: 'column', alignItems: 'center',
        gap: 12,
      }}>
        <div style={{
          width: 64, height: 64, borderRadius: 16,
          background: 'rgba(255,255,255,0.2)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 32,
        }}>⚠️</div>

        <div style={{ color: 'rgba(255,255,255,0.8)', fontSize: 13, fontWeight: 600, letterSpacing: 1 }}>
          FLOOD ALERT · 침수경보
        </div>

        <div style={{ color: 'white', fontSize: 26, fontWeight: 800, textAlign: 'center', lineHeight: 1.3 }}>
          현재 위치가 침수구역이에요
        </div>

        <div style={{ color: 'rgba(255,255,255,0.85)', fontSize: 14, textAlign: 'center' }}>
          {weather.location} · 침수심 약 {weather.floodDepth}cm 예상
        </div>
      </div>

      {/* 안내 사항 */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '20px 20px 0' }}>
        <div style={{ fontSize: 13, color: '#9CA3AF', fontWeight: 600, marginBottom: 12 }}>
          지금 꼭 지켜주세요
        </div>

        {[
          { icon: '🚗', text: '자동차 운전 금지' },
          { icon: '🌊', text: '하천변 · 지하공간 피하기' },
          { icon: '🚶', text: '물이 무릎 이상이면 즉시 대피' },
          { icon: '⚡', text: '전기차단기 · 가스밸브 잠그기' },
        ].map((item, i) => (
          <div key={i} style={{
            display: 'flex', alignItems: 'center', gap: 14,
            padding: '16px 0',
            borderBottom: i < 3 ? '1px solid #F3F4F6' : 'none',
          }}>
            <div style={{
              width: 44, height: 44, borderRadius: 12,
              background: '#FEF2F2',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 22, flexShrink: 0,
            }}>{item.icon}</div>
            <span style={{ fontSize: 16, fontWeight: 600, color: '#111' }}>{item.text}</span>
          </div>
        ))}
      </div>

      {/* 버튼 */}
      <div style={{ padding: '16px 20px 40px', display: 'flex', flexDirection: 'column', gap: 10 }}>
        <button
          onClick={onFindShelter}
          style={{
            padding: '16px', background: '#DC2626', color: 'white',
            border: 'none', borderRadius: 14, fontSize: 16, fontWeight: 700,
            cursor: 'pointer', display: 'flex', alignItems: 'center',
            justifyContent: 'center', gap: 8,
          }}
        >
          🛟 가까운 대피소 조회
        </button>

        <button
          onClick={onFindRoute}
          style={{
            padding: '16px', background: 'white', color: '#3B82F6',
            border: '1.5px solid #E5E7EB', borderRadius: 14,
            fontSize: 16, fontWeight: 700, cursor: 'pointer',
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
          }}
        >
          🧭 목적지까지 안전경로 조회
        </button>

        <button
          onClick={onClose}
          style={{
            padding: '12px', background: 'transparent', color: '#9CA3AF',
            border: 'none', fontSize: 15, cursor: 'pointer', fontWeight: 500,
          }}
        >
          닫기
        </button>
      </div>
    </div>
  );
};

export default FloodAlert;
