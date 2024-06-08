import json
import pandas as pd
import geopandas as gpd
from tqdm import *

gdf = gpd.read_file(
    "/home/tangyinzhou/inter_crime_pred/data/CHI/community.geojson"
)  # 替换为你的GeoJSON文件路径

data = pd.DataFrame({
    "index": range(len(gdf)),
    "name": gdf['community']
})
data.to_csv('/home/tangyinzhou/inter_crime_pred/data/CHI/index2name.csv', index = False)