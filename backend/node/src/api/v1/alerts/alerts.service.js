// api/v1/alerts/alerts.service.js
import { getCache } from '../../../services/redis.service.js';

/**
 * scheduler.service.js가 1분마다 'alerts:latest' 키에 저장해 둔 캐시 읽어 반환
 */
export async function getAlerts(lat, lon, limit) {
  const raw    = await getCache('alerts:latest');
  const all    = Array.isArray(raw) ? raw : [];

  // 24시간 이내 필터
  const now    = Date.now();
  const fresh  = all.filter(a => {
    const issued = new Date(a.issuedAt).getTime();
    return now - issued < 24 * 60 * 60 * 1000;
  });

  // 최신순 정렬 후 limit 적용
  const sorted = fresh
    .sort((a, b) => new Date(b.issuedAt) - new Date(a.issuedAt))
    .slice(0, limit);

  return {
    total:  sorted.length,
    alerts: sorted,
  };
}