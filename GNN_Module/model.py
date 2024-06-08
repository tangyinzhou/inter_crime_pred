import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F


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
