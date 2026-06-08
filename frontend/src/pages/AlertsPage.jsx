import { useEffect, useState } from 'react';
import { MOCK_ALERTS } from '../data/mockData';
import { getAlerts } from '../services/api';

const USE_MOCK = true;

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
  const [alerts, setAlerts]       = useState(MOCK_ALERTS);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (USE_MOCK) return;
    const timer = setTimeout(() => {
      setIsLoading(true);
      getAlerts(userLocation?.lat, userLocation?.lng)
        .then(data => setAlerts(data.alerts ?? []))
        .catch(() => {})
        .finally(() => setIsLoading(false));
    }, 0);

    return () => clearTimeout(timer);
  }, [userLocation?.lat, userLocation?.lng]);

  return (
    <div style={{ height: '100%', overflowY: 'auto', background: '#F8FAFC' }}>
      <div style={{ padding: 'max(72px, calc(env(safe-area-inset-top, 0px) + 28px)) 16px 0' }}>
        <div style={{ fontSize: 22, fontWeight: 800, color: '#111', marginBottom: 4 }}>경보</div>
        <div style={{ fontSize: 13, color: '#6B7280', marginBottom: 16 }}>
          24시간 이내 발령된 경보
        </div>
      </div>

      <div style={{ padding: '0 16px 32px', display: 'flex', flexDirection: 'column', gap: 10 }}>
        {isLoading ? (
          <div style={{ textAlign: 'center', padding: '48px 0', color: '#9CA3AF', fontSize: 14 }}>
            불러오는 중…
          </div>
        ) : alerts.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '48px 0', color: '#9CA3AF', fontSize: 14 }}>
            경보가 없어요
          </div>
        ) : (
          alerts.map((alert) => {
            return (
            <div key={alert.id} style={{
              background: 'white',
              border: '1.5px solid #FCA5A5',  // ← 단일 색상으로
              borderRadius: 16, padding: '16px',
              boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
            }}>
  
          <div style={{ fontSize: 12, color: '#9CA3AF', marginBottom: 8 }}>
            {formatTime(alert.issuedAt)}
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
