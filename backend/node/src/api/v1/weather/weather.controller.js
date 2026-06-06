import { findWeatherByLatLon } from './weather.service.js';

export async function getWeatherByLocation(req, res, next) {
  try {
    const { lat, lon } = req.query;

    if (!lat || !lon) {
      return res.status(400).json({
        message: 'lat, lon 값이 필요합니다.',
      });
    }

    const latitude = Number(lat);
    const longitude = Number(lon);

    if (Number.isNaN(latitude) || Number.isNaN(longitude)) {
      return res.status(400).json({
        message: 'lat, lon 값은 숫자여야 합니다.',
      });
    }

    const weather = await findWeatherByLatLon(latitude, longitude);

    if (!weather) {
      return res.status(404).json({
        message: '해당 위치의 날씨 데이터를 찾을 수 없습니다.',
      });
    }

    return res.status(200).json({
      gridId: weather.grid_id,
      tmfc: weather.tmfc,

      current: {
        rainfall: weather.rn1_now,
        temperature: weather.t1h_now,
        windDirection: weather.vec_now,
        windSpeed: weather.wsd_now,
      },

      forecast: {
        rainfall1h: weather.rn1_1h,
        rainfall2h: weather.rn1_2h,
        rainfall3h: weather.rn1_3h,
        rainfall4h: weather.rn1_4h,
        rainfall5h: weather.rn1_5h,
        rainfall6h: weather.rn1_6h,
        sky1h: weather.sky_1h,
      },
    });
  } catch (error) {
    next(error);
  }
}