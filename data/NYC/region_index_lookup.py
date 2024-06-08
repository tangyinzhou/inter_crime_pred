import json
import pandas as pd
import geopandas as gpd
from tqdm import *

gdf = gpd.read_file(
    "/home/tangyinzhou/inter_crime_pred/data/NYC/community.geojson"
)  # 替换为你的GeoJSON文件路径
data = pd.DataFrame({
    "area": gdf['index'],
    "name": gdf['NTAName']
})
data.to_csv('/home/tangyinzhou/inter_crime_pred/data/NYC/index2name.csv', index= False)