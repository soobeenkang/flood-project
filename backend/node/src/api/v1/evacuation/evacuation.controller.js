import * as routeService from './evacuation.service.js';

export async function getEvacuationRoute(req, res, next) {
  try {
    const startLat = parseFloat(req.query.startLat);
    const startLon = parseFloat(req.query.startLon);
    const endLat   = parseFloat(req.query.endLat);
    const endLon   = parseFloat(req.query.endLon);

    if ([startLat, startLon, endLat, endLon].some(isNaN)) {
      return res.status(400).json({
        code: 'LOCATION_REQUIRED',
        message: 'startLat, startLon, endLat, endLon 파라미터가 필요합니다.',
      });
    }
    if (
      startLat < -90 || startLat > 90 || endLat < -90 || endLat > 90 ||
      startLon < -180 || startLon > 180 || endLon < -180 || endLon > 180
    ) {
      return res.status(400).json({
        code: 'INVALID_COORDINATES',
        message: '위도·경도 범위가 올바르지 않습니다.',
      });
    }

    const result = await routeService.findEvacuationRoute(startLat, startLon, endLat, endLon);

    if (!result) {
      return res.status(404).json({
        code: 'ROUTE_NOT_FOUND',
        message: '출발지와 목적지 사이의 경로를 찾을 수 없습니다.',
      });
    }

    // GeoJSON → 프론트엔드 형식 변환
    const feature = result.features[0];
    const props   = feature.properties;
    const coords  = feature.geometry.coordinates; // [[lon, lat], ...]

    const distanceM   = props.distanceM;
    const totalMinutes = Math.round(distanceM / 80);  // 도보 속도 ~80m/min

    // coordinates → waypoints 변환 (kakao는 {lat, lon} 형식 필요)
    const waypoints = coords.map(([lon, lat]) => ({ lat, lon }));

    return res.json({
      totalMinutes,
      totalDistance:   distanceM,
      avoidedGrids:    props.hasFloodedSegment ? 1 : 0,  // 또는 별도 카운트 필요시 서비스에서 반환
      waypoints,
    });
  } catch (err) {
    next(err);
  }
}