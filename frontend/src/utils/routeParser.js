export function parseRoute(data) {
  if (!data?.routes?.length) return [];
  const linePath = [];
  data.routes[0].sections.forEach((section) => {
    section.roads?.forEach((road) => {
      const v = road.vertexes ?? [];
      for (let i = 0; i < v.length; i += 2) {
        const lng = Number(v[i]), lat = Number(v[i + 1]);
        if (!isNaN(lng) && !isNaN(lat)) linePath.push({ lat, lng });
      }
    });
  });
  return linePath.filter((pt, i, arr) =>
    i === 0 || pt.lat !== arr[i-1].lat || pt.lng !== arr[i-1].lng
  );
}
