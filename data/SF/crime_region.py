import geopandas as gpd
import pandas as pd
from tqdm import *

# 读取GeoJSON文件
gdf = gpd.read_file(
    "/home/tangyinzhou/inter_crime_pred/data/SF/community.geojson"
)  # 替换为你的GeoJSON文件路径

for year in trange(2006, 2024):
    # 假设你的DataFrame叫做df，并且包含'x'和'y'两列，分别代表横纵坐标
    # 读取DataFrame
    df = pd.read_csv(
        "/home/tangyinzhou/inter_crime_pred/data/SF/crime_{0}.csv".format(year)
    )  # 替换为你的数据文件路径

    # 将DataFrame中的坐标转换为几何点
    points = gpd.points_from_xy(df["lat"], df["lon"])
    # 将DataFrame转换为GeoDataFrame
    points_gdf = gpd.GeoDataFrame(df, geometry=points)

    # 使用sjoin方法（空间连接）来找出每个点落在哪个多边形内
    # 这将返回一个新的GeoDataFrame，其中包含原始点和它们所在的多边形
    result_gdf = gpd.sjoin(points_gdf, gdf, how="inner", op="within")
    cols = result_gdf["index"]
    df["area"] = cols
    # 打印结果
    df = df.dropna()
    df.to_csv("/home/tangyinzhou/inter_crime_pred/data/SF/crime_{0}.csv".format(year),index = False)
