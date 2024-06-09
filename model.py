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


class TGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, K):
        super(TGCNLayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.K = K
        self.weight = nn.Parameter(torch.Tensor(K, in_feats, out_feats))
        self.bias = nn.Parameter(torch.Tensor(out_feats))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, g, h):
        def chebyshev_polynomials(x, k):
            if k == 0:
                return torch.ones(size=x.size(0), 1)
            elif k == 1:
                return x
            else:
                x0 = torch.ones(size=x.size(0), 1)
                x1 = x
                for i in range(2, k+1):
                    x_next = 2 * x1.matmul(x) - x0
                    x0, x1 = x1, x_next
                return x_next

        # Compute the Chebyshev polynomials up to K for the normalized adjacency matrix.
        g = dgl.add_self_loop(g)
        adj = g.adjacency_matrix().cuda() if h.is_cuda else g.adjacency_matrix()
        deg = torch.pow(g.in_degrees().float(), -0.5)
        deg = torch.where(deg == float('inf'), torch.tensor(0.0).cuda() if h.is_cuda else torch.tensor(0.0), deg)
        norm_adj = deg.unsqueeze(1) * adj * deg.unsqueeze(0)
        A_k = chebyshev_polynomials(norm_adj, self.K)

        h = torch.matmul(A_k, h)
        h = torch.matmul(h, self.weight) + self.bias
        h = F.relu(h)

        return h

class TGCN(nn.Module):
    def __init__(self, num_nodes, history_length, feature_num, num_filters, K, num_layers, output_feature):
        super(TGCN, self).__init__()
        self.num_nodes = num_nodes
        self.history_length = history_length
        self.feature_num = feature_num
        self.num_filters = num_filters
        self.K = K
        self.num_layers = num_layers
        self.output_feature = output_feature

        self.layers = nn.ModuleList()
        self.layers.append(TGCNLayer(feature_num, num_filters, K))
        for _ in range(1, num_layers):
            self.layers.append(TGCNLayer(num_filters, num_filters, K))

        self.regressor = nn.Linear(num_filters, output_feature)

    def forward(self, g, h):
        h = h.view(-1, self.feature_num)  # Flatten the feature matrix
        for layer in self.layers:
            h = layer(g, h)
        output = self.regressor(h)
        return output.view(self.num_nodes, -1, self.output_feature)