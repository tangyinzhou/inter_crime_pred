import torch
from torch_geometric.nn import GCNConv
from torch.nn import LSTMCell, Linear


class TGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_nodes, seq_len):
        super(TGCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.seq_len = seq_len

        # 图卷积层
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

        # 循环层
        self.lstm = LSTMCell(hidden_dim, hidden_dim)

        # 输出层
        self.output_layer = Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        # x: (seq_len, num_nodes, input_dim)
        # edge_index: 图的边的索引
        x = x.permute(1, 0, 2)
        # 初始化LSTM的隐藏状态和细胞状态
        h = torch.zeros((x.size(1), self.hidden_dim)).to(x.device)
        c = torch.zeros((x.size(1), self.hidden_dim)).to(x.device)

        # 存储时间序列的输出
        seq_outputs = []

        for t in range(self.seq_len):
            # 图卷积层
            x_t = x[t]  # (num_nodes, input_dim)
            x_t = self.conv1(x_t, edge_index)
            x_t = torch.relu(x_t)

            # 循环层
            # x_t = x_t.unsqueeze(0)  # (1, num_nodes, hidden_dim)
            h, c = self.lstm(x_t, (h, c))

            # 存储当前时间步的输出
            seq_outputs.append(h)

        # 取最后一个时间步的输出
        last_output = seq_outputs[-1]

        # 通过输出层
        final_output = self.output_layer(last_output)

        return final_output
