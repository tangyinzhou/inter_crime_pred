import geopandas as gpd
import matplotlib.pyplot as plt


gdf = gpd.read_file(
    "/home/tangyinzhou/inter_crime_pred/visulization/heatmap_data.geojson"
)
gdf.plot(column="error", legend=True, cmap="OrRd")
plt.savefig("/home/tangyinzhou/inter_crime_pred/visulization/CHI_heatmap.png")
