import geopandas as gpd
import matplotlib.pyplot as plt


gdf = gpd.read_file( "inter_crime_pred/visulization/heatmap_data_CHI_our.geojson")
gdf.plot(column="error", legend=True, cmap="OrRd", vmin=10, vmax=100)
plt.savefig("inter_crime_pred/visulization/CHI_heatmap_our.png")
