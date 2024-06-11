import numpy as np
import matplotlib.pyplot as plt

preds = np.load("/home/tangyinzhou/inter_crime_pred/visulization/T_GCN_preds.npy")
pred_labels = np.load(
    "/home/tangyinzhou/inter_crime_pred/visulization/ground_truth.npy"
)
x = np.arange(len(preds))
fig, ax = plt.subplots()
ax.plot(x, preds, label="T-GCN", marker="o")  # 使用圆圈标记数据点
ax.plot(x, pred_labels, label="Ground Truth", marker="s")  # 使用正方形标记数据点
ax.legend()
ax.set_xlabel("time")
ax.set_ylabel("crime number")
ax.grid(True)
plt.savefig("/home/tangyinzhou/inter_crime_pred/visulization/CHI_line.png")
