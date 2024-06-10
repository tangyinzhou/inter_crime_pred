from data_loader import *
from TGCN_model import *
from tqdm import *
from metric_util import *

use_dataset = "CHI"
adj = load_graph(dataset=use_dataset)
input_dim, output_dim, hidden_dim, num_nodes = get_hyperparams(use_dataset)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
_, _, test_dataset = split_dataset(dataset=use_dataset)

rows, cols = np.where(adj != 0)
edge_index = torch.from_numpy(np.vstack((rows, cols))).to(device)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
# model = GCN(in_feats=31 * 11, hidden_size=64, out_feats=30)
model = TGCN(
    input_dim=input_dim - 1,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    num_nodes=adj.shape[0],
    seq_len=11,
)
model = model.to(device)
model.eval()  # 将模型设置为评估模式
total_loss = 0
preds = []
pred_labels = []
with torch.no_grad():  # 在测试阶段不计算梯度
    for features, labels, _ in test_dataloader:
        features = torch.FloatTensor(features.float()).to(device)
        labels = torch.FloatTensor(labels.float()).to(device)
        features = features.permute(0, 2, 1, 3)
        # 前向传播
        with torch.no_grad():  # 假设features是静态的，不需要梯度
            h = features
        outputs = []
        for input in h:
            pred = model(input, edge_index)
            outputs.append(pred)
        outputs = torch.stack(outputs, dim=0)
        preds.append(outputs)
        pred_labels.append(labels)
preds = torch.stack(preds, dim=0)
pred_labels = torch.stack(pred_labels, dim=0)
RMSE = cal_rmse(preds, pred_labels)
print(RMSE)
area_rmse = []
for index in range(preds.shape[2]):
    area_rmse.append(
        cal_rmse(preds[:, :, index, :], pred_labels[:, :, index, :])
        .detach()
        .cpu()
        .numpy()
    )
draw_heatmap(use_dataset, area_rmse)
