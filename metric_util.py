import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def cal_rmse(pred: np.ndarray, label: np.ndarray):
    assert pred.shape == label.shape, "Predictions and targets must have the same shape"

    # 计算误差的平方
    squared_errors = (pred - label) ** 2

    # 计算平均误差的平方（MSE）
    mean_squared_error = squared_errors.mean()

    # 计算RMSE
    rmse = mean_squared_error.sqrt()
    return rmse


def draw_heatmap(dataset: str, errors: list):
    # 加载GeoJSON数据
    gdf = gpd.read_file(
        "/home/tangyinzhou/inter_crime_pred/data/{0}/community.geojson".format(dataset)
    )
    i2n = pd.read_csv("/home/tangyinzhou/inter_crime_pred/data/CHI/index2name.csv")
    # 确保你的GeoJSON数据中有一个列来表示社区的ID或者名称
    # 例如：gdf['community_id']
    area_dict = {}
    for index, e in enumerate(errors):
        area_dict[i2n[i2n["index"] == index]["name"].values[0]] = float(e)
    # 将预测误差添加到GeoDataFrame中
    # 假设errors列表的索引与GeoDataFrame中的社区ID对应
    gdf["error"] = [area_dict[i] for i in gdf["community"]]
    gdf.plot(column="error", legend=True, cmap="OrRd")
    plt.savefig("/home/tangyinzhou/inter_crime_pred/fig_save/{0}.png".format(dataset))


def draw_temporal_fig(dataset: str, preds: np.array, pred_labels: np.array):
    preds = preds.detach().cpu().numpy()
    pred_labels = pred_labels.detach().cpu().numpy()
    preds = np.sum(preds, axis=-1)
    preds = np.sum(preds, axis=-1)
    preds = np.sum(preds, axis=-1)
    pred_labels = np.sum(pred_labels, axis=-1)
    pred_labels = np.sum(pred_labels, axis=-1)
    pred_labels = np.sum(pred_labels, axis=-1)
    x = np.arange(len(preds))
    fig, ax = plt.subplots()
    ax.plot(x, preds, label="T-GCN", marker="o")  # 使用圆圈标记数据点
    ax.plot(x, pred_labels, label="Ground Truth", marker="s")  # 使用正方形标记数据点
    ax.legend()
    ax.set_xlabel("time")
    ax.set_ylabel("crime number")
    ax.grid(True)
    plt.savefig(
        "/home/tangyinzhou/inter_crime_pred/fig_save/{0}_line.png".format(dataset)
    )
