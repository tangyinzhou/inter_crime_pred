from shapely.geometry import shape
import geopandas as gpd
import json
import pandas as pd

gdf = gpd.read_file("/home/tangyinzhou/inter_crime_pred/data/NYC/community.geojson")

# 将几何数据转换为shapely对象
gdf["geometry"] = gdf["geometry"].apply(shape)

# 初始化一个空列表来存储相邻对
community1 = []
community2 = []

# 遍历GeoDataFrame的每一行
for i, row1 in gdf.iterrows():
    # 检查除了当前行之外的每一行
    for j, row2 in gdf.loc[gdf.index != i].iterrows():
        # 使用shapely的intersects方法判断两个MULTIPOLYGON是否相交
        if row1["geometry"].intersects(row2["geometry"]):
            # 输出或存储相邻对
            community1.append(row1["NTAName"])
            community2.append(row2["NTAName"])

data = pd.DataFrame({"Community1": community1, "Community2": community2})

data.to_csv(
    "/home/tangyinzhou/inter_crime_pred/data/NYC/NYC_community_neighbor.csv",
    index=False,
)
