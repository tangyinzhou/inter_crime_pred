from data_loader import *
from model import *

G = load_graph(dataset="CHI")
train_dataloader, val_dataloader, test_dataloader = split_dataset(dataset="CHI")
model = GCN(in_feats=30 * 11, hidden_size=64, out_feats=30)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(100):  # 假设训练100个epoch
    for features, labels in train_dataloader:
        features = torch.FloatTensor(features.float())
        labels = torch.FloatTensor(labels.float())
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
