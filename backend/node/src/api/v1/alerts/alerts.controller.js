import * as svc from './alerts.service.js';

export async function list(req, res, next){
    try {
        const { lat, lon, limit = '50' } = req.query;

        const parsedLat = lat ? parseFloat(lat) : null;
        const parsedLon = lon ? parseFloat(lon) : null;
        const parsedLimit = Math.min(parseInt(limit) || 50, 100);

        if ((lat && isNaN(parsedLat)) || (lon && isNaN(parsedLon))) {
            return res.status(400).json({ code: 'INVALID_COORDINATES', message: '좌표 형식이 올바르지 않습니다.'});
        }

        const data = await svc.getAlerts(parsedLat, parsedLon, parsedLimit);
        res.json(data);
    } catch (err) {
        next(err);
    }
}