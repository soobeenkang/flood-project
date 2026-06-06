// api/v1/alerts/alerts.service.js
import redis from '../../../services/redis.service.js';
import pool from '../../../db/pool.js';

/**
 * scheduler.service.js가 1분마다 'alerts:latest' 키에 저장해 둔 캐시 읽어 반환
 */
export async function getAlerts(lat, lon, limit) {
  const raw    = await redis.get('alerts:latest');
  let all = [];

  if (raw) {
    const parsed = JSON.parse(raw);
    all = Array.isArray(parsed) ? parsed : [];
  } else {
    // 캐싱 데이터 없을 시 db에서 조회
    console.log('Redis 캐시 미스. DB에서 재난문자 조회');
    const dbResult = await pool.query(
      `SELECT source_sn AS "sourceSN", region, type, level, message, issued_at AS "issuedAt"
      FROM alerts
      WHERE issued_at >= NOW() - INTERVAL '24 hours'
      ORDER BY issued_at DESC
      LIMIT $1`,
      [limit]
    );
    return {
      total: dbResult.rows.length,
      alerts: dbResult.rows,
    };
  }

  // 24시간 이내 필터
  const now    = Date.now();
  const fresh  = all.filter(a => {
    const timeStr = a.issuedAt;
    if (!timeStr) return false;

    const utcString = timeStr.endsWith('Z') ? timeStr : `${timeStr}Z`; 
    const issued = new Date(utcString).getTime();
    return now - issued < 24 * 60 * 60 * 1000;
  });

  // 최신순 정렬 후 limit 적용
  const sorted = fresh
    .sort((a, b) => {
      const timeB = b.issuedAt || b.issuedAT;
      const timeA = a.issuedAt || a.issuedAT;
      const utcB = timeB.endsWith('Z') ? timeB : `${timeB}Z`;
      const utcA = timeA.endsWith('Z') ? timeA : `${timeA}Z`;
      return new Date(utcB) - new Date(utcA);
    })
    .slice(0, limit);

  return {
    total:  sorted.length,
    alerts: sorted,
  };
}