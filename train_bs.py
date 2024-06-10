from data_loader import *
from TGCN_model import *
from tqdm import *


def validate(dataloader, model, criterion):
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


def test(dataloader, model, criterion):
    model.eval()  # 将模型设置为评估模式
    total_loss = 0
    with torch.no_grad():  # 在测试阶段不计算梯度
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


use_dataset = "CHI"
adj = load_graph(dataset=use_dataset)
input_dim, output_dim, hidden_dim, num_nodes = get_hyperparams(use_dataset)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
train_dataset, val_dataset, test_dataset = split_dataset(dataset=use_dataset)

rows, cols = np.where(adj != 0)
edge_index = torch.from_numpy(np.vstack((rows, cols))).to(device)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
# model = GCN(in_feats=31 * 11, hidden_size=64, out_feats=30)
model = TGCN(
    input_dim=input_dim - 1,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    num_nodes=adj.shape[0],
    seq_len=11,
)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.MSELoss()
patience = 20  # 允许的连续没有改善的epoch数
counter = 0  # 连续没有改善的epoch计数器
best_loss = float("inf")  # 最佳验证损失
for epoch in trange(1000):  # 假设训练100个epoch
    for features, labels, _ in train_dataloader:
        model.train()
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

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item()}")
    val_loss = validate(val_dataloader, model, criterion)
    print(f"Epoch {epoch}, Val Loss: {val_loss}")
    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0  # 重置计数器
        # 保存最佳模型的权重
        torch.save(
            model.state_dict(), "/home/tangyinzhou/inter_crime_pred/model_save/TGCN.pth"
        )
    else:
        counter += 1
        print(f"No improvement for {counter} epochs")

    # 如果连续没有改善的epoch数达到patience，则停止训练
    if counter >= patience:
        print(f"Early stopping after {epoch+1} epochs")
        break

# test_loss = test(test_dataloader, model, criterion)
# print(f"Test Loss: {test_loss}")
