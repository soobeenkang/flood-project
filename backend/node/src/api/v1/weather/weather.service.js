import pool from '../../../db/pool.js';

export async function findWeatherByLatLon(lat, lon) {
  const { rows } = await pool.query(
    `
    WITH nearest_grid AS (
      SELECT grid_id
      FROM flood_grid
      ORDER BY
        ((center_lat - $1) * (center_lat - $1)
        + (center_lng - $2) * (center_lng - $2)) ASC
      LIMIT 1
    )
    SELECT
      w.grid_id,
      w.tmfc,

      w.rn1_now,
      w.t1h_now,
      w.vec_now,
      w.wsd_now,

      w.rn1_1h,
      w.rn1_2h,
      w.rn1_3h,
      w.rn1_4h,
      w.rn1_5h,
      w.rn1_6h,

      w.sky_1h
    FROM seoul_weather w
    JOIN nearest_grid g
      ON w.grid_id = g.grid_id
    ORDER BY w.tmfc DESC
    LIMIT 1
    `,
    [lat, lon]
  );

  return rows[0];
}