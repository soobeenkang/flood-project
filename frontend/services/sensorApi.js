import api from './api';

/**
 * 전체 센서 목록 조회
 * @returns {Promise<Array>}
 */
export const fetchSensors = () => {
  return api.get('/sensors');
};

/**
 * 특정 센서 수위 조회
 * @param {string|number} id - 센서 ID
 * @returns {Promise<Object>}
 */
export const fetchSensorById = (id) => {
  return api.get(`/sensors/${id}`);
};