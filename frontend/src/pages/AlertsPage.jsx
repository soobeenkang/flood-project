import { useEffect, useState } from 'react';
import { MOCK_ALERTS } from '../data/mockData';
import { getAlerts } from '../services/api';

const USE_MOCK = true;

const LEVEL_CONFIG = {
  '심각': { border: '#FCA5A5', badge: '#DC2626' },
  '경보': { border: '#FCD34D', badge: '#F59E0B' },
  '주의': { border: '#FDE68A', badge: '#D97706' },
};

const formatTime = (isoString) => {
  const diff = Date.now() - new Date(isoString).getTime();
  const min  = Math.floor(diff / 60000);
  const hour = Math.floor(min / 60);
  if (min < 1)   return '방금 전';
  if (min < 60)  return `${min}분 전`;
  if (hour < 24) return `${hour}시간 전`;
  return `${Math.floor(hour / 24)}일 전`;
};

const AlertsPage = ({ userLocation }) => {
  const [filter, setFilter]   = useState('all');
  const [alerts, setAlerts]   = useState(MOCK_ALERTS);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (USE_MOCK) return;
    setIsLoading(true);
    getAlerts(userLocation?.lat, userLocation?.lng)
      .then(data => setAlerts(data.alerts ?? []))
      .catch(() => {})
      .finally(() => setIsLoading(false));
  }, []);

  const FILTERS = [
    { key: 'all',  label: '전체' },
    { key: '심각', label: '심각' },
    { key: '경보', label: '경보' },
    { key: '주의', label: '주의' },
  ];

  const filtered = filter === 'all'
    ? alerts
    : alerts.filter(a => a.level === filter);

  return (
    <div style={{ height: '100%', overflowY: 'auto', background: '#F8FAFC' }}>
      <div style={{ padding: '20px 16px 0' }}>
        <div style={{ fontSize: 22, fontWeight: 800, color: '#111', marginBottom: 4 }}>경보</div>
        <div style={{ fontSize: 13, color: '#6B7280', marginBottom: 16 }}>
          24시간 이내 발령된 경보
        </div>
        <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
          {FILTERS.map((f) => (
            <button key={f.key} onClick={() => setFilter(f.key)} style={{
              padding: '8px 16px', border: 'none', borderRadius: 20,
              fontSize: 13, fontWeight: 600, cursor: 'pointer',
              background: filter === f.key ? '#EF4444' : 'white',
              color: filter === f.key ? 'white' : '#374151',
              boxShadow: filter === f.key
                ? '0 2px 8px rgba(239,68,68,0.3)'
                : '0 1px 3px rgba(0,0,0,0.08)',
            }}>{f.label}</button>
          ))}
        </div>
      </div>

      <div style={{ padding: '0 16px 32px', display: 'flex', flexDirection: 'column', gap: 10 }}>
        {isLoading ? (
          <div style={{ textAlign: 'center', padding: '48px 0', color: '#9CA3AF', fontSize: 14 }}>
            불러오는 중…
          </div>
        ) : filtered.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '48px 0', color: '#9CA3AF', fontSize: 14 }}>
            해당 경보가 없어요
          </div>
        ) : (
          filtered.map((alert) => {
            const cfg = LEVEL_CONFIG[alert.level] ?? LEVEL_CONFIG['주의'];
            return (
              <div key={alert.id} style={{
                background: 'white',
                border: `1.5px solid ${cfg.border}`,
                borderRadius: 16, padding: '16px',
                boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                  <span style={{
                    background: cfg.badge, color: 'white',
                    fontSize: 11, fontWeight: 700,
                    padding: '2px 8px', borderRadius: 20, flexShrink: 0,
                  }}>{alert.level}</span>
                  <span style={{ fontSize: 12, color: '#9CA3AF' }}>
                    {formatTime(alert.issuedAt)}
                  </span>
                </div>
                <div style={{ fontSize: 14, color: '#374151', lineHeight: 1.6 }}>
                  {alert.message}
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};

export default AlertsPage;
