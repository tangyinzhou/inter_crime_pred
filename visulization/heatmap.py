import geopandas as gpd
import matplotlib.pyplot as plt


gdf = gpd.read_file( "inter_crime_pred/visulization/heatmap_data_fake.geojson")
gdf.plot(column="error", legend=True, cmap="OrRd")
plt.savefig("inter_crime_pred/visulization/CHI_heatmap_opt.png")
