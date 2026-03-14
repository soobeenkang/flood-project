import geopandas as gpd
import matplotlib.pyplot as plt

grid = gpd.read_file("data/grid/gangnam_grid.geojson")

fig, ax = plt.subplots(figsize=(8,8))

grid.plot(ax=ax, edgecolor="black", facecolor="none")

plt.title("Gangnam 100m Grid")

plt.show()