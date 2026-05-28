import api from './api';

/**
 * 침수 히트맵 데이터 조회
 * @param {string} t - 시간 파라미터 ('', '1h', '3h', '6h')
 * @returns {Promise<Array<{ grid_id: number, flood: number }>>}
 */
export const fetchHeatmap = (t = '') => {
  const params = t ? { t } : {};
  return api.get('/flood/heatmap', { params });
};