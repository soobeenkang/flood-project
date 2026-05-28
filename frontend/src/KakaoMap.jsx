import { useEffect, useRef } from 'react';

const KakaoMap = () => {
  const mapRef = useRef(null);

  useEffect(() => {
    const initMap = () => {
      const map = new window.kakao.maps.Map(mapRef.current, {
        center: new window.kakao.maps.LatLng(37.5665, 126.978),
        level: 7,
      });
    };

    if (window.kakao && window.kakao.maps) {
      initMap();
    } else {
      const script = document.querySelector('script[src*="dapi.kakao.com"]');
      script?.addEventListener('load', initMap);
    }
  }, []);

  return (
    <div
      ref={mapRef}
      style={{ width: '100%', height: '100vh' }}
    />
  );
};

export default KakaoMap;