function isPointInPolygon(point, vs) {
  const x = point.lng, y = point.lat;
  let inside = false;
  for (let i = 0, j = vs.length - 1; i < vs.length; j = i++) {
    const xi = vs[i][0], yi = vs[i][1];
    const xj = vs[j][0], yj = vs[j][1];
    if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) inside = !inside;
  }
  return inside;
}

export function pathCrossFlood(path, floodMap, features) {
  if (!features || !path?.length) return false;
  for (const point of path) {
    for (const feature of features) {
      const geojsonId = feature.id || feature.properties?.grid_id || feature.properties?.id;
      const flood = floodMap.get(Number(geojsonId));
      if (!flood) continue;
      let coords = feature.geometry.coordinates;
      if (feature.geometry.type === "MultiPolygon") coords = coords[0][0];
      else if (feature.geometry.type === "Polygon") coords = coords[0];
      if (!coords?.length) continue;
      if (isPointInPolygon(point, coords)) return true;
    }
  }
  return false;
}
