import { http, HttpResponse } from 'msw';
import { heatmapFixture, sensorsFixture } from './fixtures/heatmapFixture';

const BASE = 'http://localhost:3000/api/v1';

export const handlers = [
  // ── 히트맵 ───────────────────────────────────────────────────────────────
  http.get(`${BASE}/flood/heatmap`, ({ request }) => {
    const url = new URL(request.url);
    const t = url.searchParams.get('t') ?? '';

    const data = heatmapFixture[t] ?? heatmapFixture.current;

    // 실제 API 응답 속도 시뮬레이션 (200ms)
    return new Promise((resolve) =>
      setTimeout(() => resolve(HttpResponse.json(data)), 200)
    );
  }),

  // ── 센서 목록 ─────────────────────────────────────────────────────────────
  http.get(`${BASE}/sensors`, () => {
    return HttpResponse.json(sensorsFixture);
  }),

  // ── 특정 센서 ─────────────────────────────────────────────────────────────
  http.get(`${BASE}/sensors/:id`, ({ params }) => {
    const sensor = sensorsFixture.find((s) => String(s.id) === params.id);
    if (!sensor) {
      return new HttpResponse(null, { status: 404 });
    }
    return HttpResponse.json(sensor);
  }),

  // ── 대피경로 ─────────────────────────────────────────────────────────────
  http.post(`${BASE}/evacuation/route`, async ({ request }) => {
    const body = await request.json();
    const { origin } = body;

    // Mock 경로: 출발지 → 임시 대피소 (서울시청)
    return HttpResponse.json({
      path: [
        { lat: origin.lat, lng: origin.lng },
        { lat: origin.lat + 0.005, lng: origin.lng + 0.005 },
        { lat: 37.5665, lng: 126.978 },
      ],
      distance: 1200, // m
      duration: 15,   // 분
    });
  }),

  // ── 관리자 상태 ───────────────────────────────────────────────────────────
  http.get(`${BASE}/admin/status`, () => {
    return HttpResponse.json({
      lastCollected: new Date().toISOString(),
      sensorCount: sensorsFixture.length,
      modelStatus: 'ready',
    });
  }),
];