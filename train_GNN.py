from data_loader import *

# from model import *
from TGCN_model import *


def validate(dataloader, model, criterion, device, edge_index):
    model.eval()  # 将模型设置为评估模式
    total_loss = 0
    with torch.no_grad():  # 在验证阶段不计算梯度
        for features, labels, _ in dataloader:
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
            # 计算损失
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def train_GNN(
    gnn_train_dataset, gnn_val_dataset, gnn_weight, adj, hyper_params, device
):
    # 找到非零元素的索引
    rows, cols = np.where(adj != 0)
    edge_index = torch.from_numpy(np.vstack((rows, cols)))
    train_dataloader = DataLoader(gnn_train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(gnn_val_dataset, batch_size=32, shuffle=True)
    model = TGCN(
        input_dim=hyper_params["input_dim"],
        hidden_dim=hyper_params["hidden_dim"],
        output_dim=hyper_params["output_dim"],
        num_nodes=adj.shape[0],
        seq_len=11,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    patience = 20  # 允许的连续没有改善的epoch数
    counter = 0  # 连续没有改善的epoch计数器
    best_loss = float("inf")  # 最佳验证损失
    for epoch in range(1000):  # 假设训练100个epoch
        for features, labels, llm_pred in train_dataloader:
            features = torch.FloatTensor(features.float()).to(device)
            labels = torch.FloatTensor(labels.float()).to(device)
            llm_pred = torch.FloatTensor(llm_pred.float()).to(device)
            llm_pred = llm_pred.repeat(1, 11, 1, 1)
            features = torch.cat((features, llm_pred * gnn_weight), dim=-1)
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
        print(f"Epoch {epoch}, Loss: {loss.item()}")
        val_loss = validate(val_dataloader, model, criterion, 1, edge_index)
        print(f"Epoch {epoch}, Val Loss: {val_loss}")
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0  # 重置计数器
            # 保存最佳模型的权重
            torch.save(
                model.state_dict(),
                "/home/tangyinzhou/inter_crime_pred/model_save/LLM_TGCN.pth",
            )
        else:
            counter += 1
            print(f"No improvement for {counter} epochs")

        # 如果连续没有改善的epoch数达到patience，则停止训练
        if counter >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
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
