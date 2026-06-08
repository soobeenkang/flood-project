import { http, HttpResponse } from 'msw';
import { heatmapFixture, sensorsFixture } from './fixtures/heatmapFixture';

const RAW_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:3000';
const BASE = `${RAW_BASE}/api/v1`;

export const handlers = [
  // ── 히트맵 ───────────────────────────────────────────────────────────────
  http.get(`${BASE}/heatmap/grids`, ({ request }) => {
    const url = new URL(request.url);
    const t = url.searchParams.get('horizon') ?? 'now';

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
  http.post(`${BASE}/route/evacuation`, async ({ request }) => {
    const body = await request.json();
    const origin = {
      lat: body.originLat ?? body.origin?.lat,
      lng: body.originLon ?? body.origin?.lng,
    };

    // Mock 경로: 출발지 → 임시 대피소 (서울시청)
    return HttpResponse.json({
      waypoints: [
        { lat: origin.lat, lng: origin.lng },
        { lat: origin.lat + 0.005, lng: origin.lng + 0.005 },
        { lat: 37.5665, lng: 126.978 },
      ],
      totalDistance: 1200, // m
      totalMinutes: 15,    // 분
      avoidedGrids: 0,
    });
  }),

  // ── 이메일 알림 구독 ───────────────────────────────────────────────────
  http.post(`${BASE}/subscriptions`, async () => {
    return HttpResponse.json({ ok: true }, { status: 201 });
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
