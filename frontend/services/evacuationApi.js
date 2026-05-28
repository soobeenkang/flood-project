import api from './api';

/**
 * 대피경로 요청
 * @param {{ origin: { lat: number, lng: number }, destination?: { lat: number, lng: number } }} payload
 * @returns {Promise<Object>}
 */
export const fetchEvacuationRoute = (payload) => {
  return api.post('/evacuation/route', payload);
};