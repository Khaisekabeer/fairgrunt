"""FairGRUNT-T network: residual GCN encoder, MLP encoder and two prediction heads."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """GCN layer (Kipf & Welling, 2017): A_hat @ X @ W + b."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.xavier_normal_(self.weight)
        bound = 1.0 / math.sqrt(out_features)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, adj):
        return torch.spmm(adj, x @ self.weight) + self.bias


class ResidualGCN(nn.Module):
    """Three GCN layers with BatchNorm and a residual link around the second layer."""

    def __init__(self, n_in, n_hidden, n_out, dropout):
        super().__init__()
        self.gc1, self.bn1 = GraphConvolution(n_in, n_hidden), nn.BatchNorm1d(n_hidden)
        self.gc2, self.bn2 = GraphConvolution(n_hidden, n_hidden), nn.BatchNorm1d(n_hidden)
        self.gc3, self.bn3 = GraphConvolution(n_hidden, n_out), nn.BatchNorm1d(n_out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(F.relu(self.bn1(self.gc1(x, adj))), self.dropout, self.training)
        x = F.dropout(F.relu(self.bn2(self.gc2(x, adj))), self.dropout, self.training) + x
        return self.bn3(self.gc3(x, adj))


def mlp(sizes, dropout):
    """Linear layers with ReLU and dropout between them (none after the last layer)."""
    layers = []
    for i in range(len(sizes) - 2):
        layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    return nn.Sequential(*layers)


class FairGRUNTT(nn.Module):
    """h_v = MLP(GCN(X, A_hat)) is split into z_s (cohort) and z_c (task).

    The fraud head sees only z_c and the cohort head sees only z_s.
    """

    def __init__(self, n_features, cfg):
        super().__init__()
        self.cohort_dim = cfg.cohort_dim
        task_dim = cfg.emb_dim - cfg.cohort_dim
        # Construction order matters for reproducing the reported runs (same RNG draws).
        self.gcn = ResidualGCN(n_features, cfg.gcn_hidden, cfg.emb_dim, cfg.gcn_dropout)
        self.encoder = mlp([cfg.emb_dim, cfg.mlp_hidden, cfg.emb_dim], cfg.mlp_dropout)
        self.task_head = mlp([task_dim, *cfg.task_head, 2], cfg.mlp_dropout)
        self.cohort_head = mlp([cfg.cohort_dim, *cfg.cohort_head, 2], cfg.mlp_dropout)

    def embed(self, x, adj):
        h = self.encoder(self.gcn(x, adj))
        return h[:, :self.cohort_dim], h[:, self.cohort_dim:], h  # z_s, z_c, h

    @torch.no_grad()
    def fraud_probability(self, x, adj):
        _, z_c, _ = self.embed(x, adj)
        return F.softmax(self.task_head(z_c), dim=1)[:, 1]
