const locationService = require('./location.service');

async function floodCheck(req, res, next) {
  try {
    const lat = parseFloat(req.query.lat);
    const lon = parseFloat(req.query.lon);

    if (isNaN(lat) || isNaN(lon)) {
      return res.status(400).json({
        code: 'LOCATION_REQUIRED',
        message: 'lat, lon 파라미터가 필요합니다.',
      });
    }
    if (lat < -90 || lat > 90 || lon < -180 || lon > 180) {
      return res.status(400).json({
        code: 'INVALID_COORDINATES',
        message: '위·경도 범위가 올바르지 않습니다.',
      });
    }

    const result = await locationService.checkFloodAtPoint(lat, lon);
    return res.json(result);
  } catch (err) {
    next(err);
  }
}

module.exports = { floodCheck };