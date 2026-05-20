INSERT INTO road_edge_grid (edge_id, grid_id)
SELECT
    r.edge_id,
    g.grid_id
FROM road_edge r
JOIN flood_grid g
ON ST_Intersects(r.geom, g.geom)
ON CONFLICT DO NOTHING;
