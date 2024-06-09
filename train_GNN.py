from data_loader import *
from model import *

# G = load_graph(dataset="CHI")
# train_dataloader, val_dataloader, test_dataloader = split_dataset(dataset="CHI")
# model = GCN(in_feats=30 * 11, hidden_size=64, out_feats=30)
# # model = TGCN(in_feats=30, hidden_size=64, out_feats=30, num_layers=3, K=11)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# criterion = nn.MSELoss()

# for epoch in range(100):  # 假设训练100个epoch
#     for features, labels, _ in train_dataloader:
#         features = torch.FloatTensor(features)
#         labels = torch.FloatTensor(labels.float())
#         features = features.permute(0, 2, 1, 3)
#         # 前向传播
#         with torch.no_grad():  # 假设features是静态的，不需要梯度
#             h = features
#         pred = model(G, h)

#         # 计算损失
#         loss = criterion(pred, labels)

#         # 反向传播和优化
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     print(f"Epoch {epoch+1}, Loss: {loss.item()}")


def train_GNN(gnn_train_dataset, gnn_val_dataset, gnn_weight, G):
    G = load_graph(dataset="CHI")
    train_dataloader = DataLoader(gnn_train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(gnn_val_dataset, batch_size=32, shuffle=True)
    model = GCN(in_feats=31 * 11, hidden_size=64, out_feats=30)
    # model = TGCN(in_feats=30, hidden_size=64, out_feats=30, num_layers=3, K=11)
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
            pred = model(G, h)

            # 计算损失
            loss = criterion(pred, labels)

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
