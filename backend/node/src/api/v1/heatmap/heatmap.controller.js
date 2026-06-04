// api/v1/heatmap/heatmap.controller.js
import * as svc from './heatmap.service.js';

const VALID_HORIZONS = new Set(['now', '1h', '3h', '6h']);

function parseLocation(req, res) {
    const lat = parseFloat(req.query.lat);
    const lon = parseFloat(req.query.lon);
   
    if (isNaN(lat) || isNaN(lon)) {
      res.status(400).json({ code: 'LOCATION_REQUIRED', message: 'lat, lon 파라미터가 필요합니다.' });
      return null;
    }
    if (lat < -90 || lat > 90 || lon < -180 || lon > 180) {
      res.status(400).json({ code: 'INVALID_COORDINATES', message: '위·경도 범위가 올바르지 않습니다.' });
      return null;
    }
    return { lat, lon };
  }


// GET /api/v1/heatmap/grids?lat=&lon=&radius=&horizon=
export async function getGrids(req, res, next) {
  try {
    const coords = parseLocation(req, res);
    if (!coords) return;
 
    const radius  = Math.min(Number(req.query.radius) || 2000, 10000);
    const horizon = VALID_HORIZONS.has(req.query.horizon) ? req.query.horizon : 'now';
 
    const result = await svc.getGrids(coords.lat, coords.lon, radius, horizon);
    return res.json(result);
  } catch (err) {
    next(err);
  }
}


// GET /api/v1/heatmap/grids/all-horizons?lat=&lon=&radius=
export async function getAllHorizons(req, res, next) {
  try {
    const coords = parseLocation(req, res);
    if (!coords) return;
 
    const radius = Math.min(Number(req.query.radius) || 2000, 10000);
 
    const result = await svc.getAllHorizons(coords.lat, coords.lon, radius);
    return res.json(result);
  } catch (err) {
    next(err);
  }
}


// GET /api/v1/heatmap/grids/:id
export async function getGrid(req, res, next) {
  try {
    const { id } = req.params;
 
    if (!/^\d+$/.test(id)) {
      return res.status(400).json({ code: 'INVALID_COORDINATES', message: '올바르지 않은 grid id 형식입니다.' });
    }
 
    const detail = await svc.getGrid(id);
    if (!detail) {
      return res.status(404).json({ code: 'GRID_NOT_FOUND', message: '해당 그리드를 찾을 수 없습니다.' });
    }
 
    return res.json(detail);
  } catch (err) {
    next(err);
  }
}