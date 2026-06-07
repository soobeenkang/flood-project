import pool from "../../../db/pool.js";

const getWalkMinutes = (distance) => {
  return Math.ceil(distance / 67); // 보행 속도 약 4km/h
};

export const getShelters = async (query) => {
  const lat = Number(query.lat);
  const lon = Number(query.lon);
  const radius = Number(query.radius || 3000);

  if (Number.isNaN(lat) || Number.isNaN(lon)) {
    throw new Error("lat, lon은 필수입니다.");
  }

  const { rows } = await pool.query(
    `
    SELECT
      shelter_id AS id,
      name,
      address,
      type,
      ST_Y(geom) AS lat,
      ST_X(geom) AS lon,
      ST_DistanceSphere(
        geom,
        ST_SetSRID(ST_MakePoint($2, $1), 4326)
      ) AS distance
    FROM shelter
    WHERE ST_DWithin(
      geom::geography,
      ST_SetSRID(ST_MakePoint($2, $1), 4326)::geography,
      $3
    )
    ORDER BY distance ASC
    `,
    [lat, lon, radius]
  );

  const shelters = rows.map((shelter) => ({
    id: shelter.id,
    name: shelter.name,
    address: shelter.address,
    type: shelter.type,
    lat: Number(shelter.lat),
    lon: Number(shelter.lon),
    distance: Math.round(Number(shelter.distance)),
    walkMinutes: getWalkMinutes(Number(shelter.distance)),
  }));

  return { shelters };
};

export const getShelterDetail = async (shelterId) => {
  const { rows } = await pool.query(
    `
    SELECT
      shelter_id AS id,
      name,
      address,
      type,
      ST_Y(geom) AS lat,
      ST_X(geom) AS lon
    FROM shelter
    WHERE shelter_id = $1
    `,
    [shelterId]
  );

  if (rows.length === 0) {
    throw new Error("대피소를 찾을 수 없습니다.");
  }

  return {
    id: rows[0].id,
    name: rows[0].name,
    address: rows[0].address,
    type: rows[0].type,
    lat: Number(rows[0].lat),
    lon: Number(rows[0].lon),
  };
};