import { useEffect } from "react";
import { SEOUL_GRID_GEOJSON_URL } from "../../constants/mapConfig";

function getPolygonPaths(geometry) {
  const { type, coordinates } = geometry;

  if (type === "Polygon") {
    return coordinates.map((ring) =>
      ring.map(([lng, lat]) => new window.kakao.maps.LatLng(lat, lng))
    );
  }

  if (type === "MultiPolygon") {
    return coordinates.flatMap((polygon) =>
      polygon.map((ring) =>
        ring.map(([lng, lat]) => new window.kakao.maps.LatLng(lat, lng))
      )
    );
  }

  return [];
}

export default function SeoulGridLayer({ map }) {
  useEffect(() => {
    if (!map || !window.kakao) return;

    const polygons = [];

    async function drawGrid() {
      const res = await fetch(SEOUL_GRID_GEOJSON_URL);
      const geojson = await res.json();

      geojson.features.forEach((feature) => {
        const paths = getPolygonPaths(feature.geometry);

        paths.forEach((path) => {
          const polygon = new window.kakao.maps.Polygon({
            path,
            strokeWeight: 1,
            strokeColor: "#2563eb",
            strokeOpacity: 0.35,
            fillColor: "#3b82f6",
            fillOpacity: 0.12,
          });

          polygon.setMap(map);
          polygons.push(polygon);
        });
      });
    }

    drawGrid();

    return () => {
      polygons.forEach((polygon) => polygon.setMap(null));
    };
  }, [map]);

  return null;
}