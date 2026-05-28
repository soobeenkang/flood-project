import { useState, useEffect } from 'react';
import { MAP_CONFIG } from '../constants/mapConfig';

/**
 * useGeolocation
 *
 * 브라우저 Geolocation API로 현재 위치 반환
 * 권한 거부 또는 실패 시 서울 기본 좌표로 fallback
 */
const useGeolocation = () => {
  const [location, setLocation] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!navigator.geolocation) {
      setLocation(MAP_CONFIG.center);
      setIsLoading(false);
      return;
    }

    navigator.geolocation.getCurrentPosition(
      (position) => {
        setLocation({
          lat: position.coords.latitude,
          lng: position.coords.longitude,
        });
        setIsLoading(false);
      },
      (err) => {
        console.warn('[useGeolocation] 위치 권한 거부 또는 오류, 기본 좌표 사용:', err.message);
        setError(err.message);
        setLocation(MAP_CONFIG.center); // 서울 시청 fallback
        setIsLoading(false);
      },
      {
        timeout: 5000,
        maximumAge: 60000,
      }
    );
  }, []);

  return { location, isLoading, error };
};

export default useGeolocation;