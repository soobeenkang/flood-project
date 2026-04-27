import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx

grid = gpd.read_file("data/grid/seoul_grid.geojson")

grid_zoom = grid.cx[127.02:127.05, 37.49:37.52]
grid_zoom = grid_zoom.to_crs(epsg=3857)

fig, ax = plt.subplots(figsize=(8,8))

grid_zoom.plot(ax=ax, edgecolor="black", facecolor="none", linewidth=0.3)

ctx.add_basemap(ax)

plt.show()