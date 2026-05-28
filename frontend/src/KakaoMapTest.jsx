import { useEffect, useRef } from "react";

export default function KakaoMapTest() {
  const mapRef = useRef(null);

  useEffect(() => {
    const kakaoKey = import.meta.env.VITE_KAKAO_MAP_KEY;

    console.log("KAKAO KEY:", kakaoKey);

    const script = document.createElement("script");

    script.src = `https://dapi.kakao.com/v2/maps/sdk.js?appkey=${kakaoKey}&autoload=false`;

    script.async = true;

    document.head.appendChild(script);

    script.onload = () => {
      window.kakao.maps.load(() => {
        const container = mapRef.current;

        const options = {
          center: new window.kakao.maps.LatLng(37.5665, 126.9780),
          level: 3,
        };

        new window.kakao.maps.Map(container, options);
      });
    };
  }, []);

  return (
    <div
      ref={mapRef}
      style={{
        width: "100vw",
        height: "100vh",
      }}
    />
  );
}