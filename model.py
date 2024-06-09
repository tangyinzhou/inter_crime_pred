import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv


class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(GCN, self).__init__()
        self.conv1 = dgl.nn.GraphConv(in_feats, hidden_size, allow_zero_in_degree=True)
        self.conv2 = dgl.nn.GraphConv(hidden_size, out_feats, allow_zero_in_degree=True)

    def forward(self, g, in_feat):
        output = []
        for input in in_feat:
            input = input.reshape(77, -1)
            h = self.conv1(g, input)
            h = torch.relu(h)
            h = self.conv2(g, h)
            output.append(h)
        output = torch.stack(output, dim=0)
        return output


class TGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(TGCNLayer, self).__init__()
        self.graph_conv = GraphConv(in_feats, out_feats)

    def forward(self, g, inputs, hidden_state):
        # 将输入和隐藏状态连接起来
        x = torch.cat([inputs, hidden_state], dim=-1)
        # 应用图卷积
        h = self.graph_conv(g, x)
        return h


class TGCNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TGCNCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.tgcn_layer = TGCNLayer(input_dim + hidden_dim, hidden_dim * 2)
        self.tgcn_update = TGCNLayer(input_dim + hidden_dim, hidden_dim)

    def forward(self, g, inputs, hidden_state):
        # 通过图卷积层计算候选隐藏状态
        concatenated = self.tgcn_layer(g, inputs, hidden_state)
        # 分离重置门和更新门
        resetgate, updategate = torch.split(concatenated, self.hidden_dim, dim=-1)
        resetgate = torch.sigmoid(resetgate)
        updategate = torch.sigmoid(updategate)
        # 计算新的候选隐藏状态
        new_cand = self.tgcn_update(inputs, resetgate * hidden_state)
        # 计算新的隐藏状态
        new_hidden_state = updategate * hidden_state + (1 - updategate) * torch.tanh(
            new_cand
        )
        return new_hidden_state, new_hidden_state


class TGCN(nn.Module):
    def __init__(self, hidden_dim):
        super(TGCN, self).__init__()
        self.hidden_dim = hidden_dim
        self.tgcn_cell = TGCNCell(hidden_dim, hidden_dim)

    def forward(self, g, inputs):
        # 假设inputs是(batch_size, seq_len, num_nodes, input_dim)
        batch_size, seq_len, num_nodes, _ = inputs.shape
        hidden_state = torch.zeros(
            batch_size, num_nodes, self.hidden_dim, device=inputs.device
        )
        outputs = []
        for t in range(seq_len):
            hidden_state, _ = self.tgcn_cell(g, inputs[:, t, :, :], hidden_state)
            outputs.append(hidden_state)
        outputs = torch.stack(
            outputs, dim=1
        )  # (batch_size, seq_len, num_nodes, hidden_dim)
        return outputs
