from data_loader import *

# from model import *
from TGCN_model import *


def train_GNN(gnn_train_dataset, gnn_val_dataset, gnn_weight, adj):
    # 找到非零元素的索引
    rows, cols = np.where(adj != 0)
    edge_index = torch.from_numpy(np.vstack((rows, cols)))
    # edge_index = edge_index.T
    train_dataloader = DataLoader(gnn_train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(gnn_val_dataset, batch_size=32, shuffle=True)
    # model = GCN(in_feats=31 * 11, hidden_size=64, out_feats=30)
    model = TGCN(
        input_dim=31,
        hidden_dim=64,
        output_dim=30,
        num_nodes=adj.shape[0],
        seq_len=11,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(2):  # 假设训练100个epoch
        for features, labels, llm_pred in train_dataloader:
            features = torch.FloatTensor(features.float())
            labels = torch.FloatTensor(labels.float())
            llm_pred = torch.FloatTensor(llm_pred.float())
            llm_pred = llm_pred.repeat(1, 11, 1, 1)
            features = torch.cat((features, llm_pred), dim=-1)
            features = features.permute(0, 2, 1, 3)
            # 前向传播
            with torch.no_grad():  # 假设features是静态的，不需要梯度
                h = features
            outputs = []
            for input in h:
                pred = model(input, edge_index)
                outputs.append(pred)
            outputs = torch.stack(outputs, dim=0)
            # 计算损失
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    return model


def gnn_pred_func(dataset, model, G):
    dataloader = DataLoader(dataset, batch_size=1)
    gnn_pred = []
    dataset_data = []
    for features, labels, llm_pred in dataloader:
        features = torch.FloatTensor(features.float())
        labels = torch.FloatTensor(labels.float())
        llm_pred = torch.FloatTensor(llm_pred.float())
        llm_pred = llm_pred.repeat(1, 11, 1, 1)
        feature_per = torch.cat((features, llm_pred), dim=-1)
        feature_per = feature_per.permute(0, 2, 1, 3)
        # 前向传播
        with torch.no_grad():  # 假设features是静态的，不需要梯度
            h = feature_per
        pred = model(G, h)
        gnn_pred.append(pred.detach().numpy())
        dataset_data.append(
            np.concatenate((features.squeeze(0).numpy(), labels.numpy()), axis=0)
        )
    gnn_pred = np.array(gnn_pred)
    dataset = LLMDataset(dataset_data, gnn_pred)
    return dataset
