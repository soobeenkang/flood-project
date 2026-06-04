const REST_API_KEY = import.meta.env.VITE_KAKAO_REST_API_KEY;

export async function getDirection(start, end, waypoint = null) {
  const url = "https://apis-navi.kakaomobility.com/v1/directions";

  const params = {
    origin: `${start.lng},${start.lat}`,
    destination: `${end.lng},${end.lat}`,
    priority: "DISTANCE",
  };

  if (waypoint) {
    params.waypoints = `${waypoint.lng},${waypoint.lat}`;
  }

  const query = new URLSearchParams(params);

  const response = await fetch(`${url}?${query}`, {
    headers: { Authorization: `KakaoAK ${REST_API_KEY}` },
  });

  if (!response.ok) throw new Error("경로 탐색 실패");
  return response.json();
}
