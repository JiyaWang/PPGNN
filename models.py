import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from utils.sparse_tensor import normal_diag, edge_index_to_sparse_tensor
from utils.anchor import sample_anchors
from torch_geometric.nn.inits import zeros
from torch_geometric.nn.dense.linear import Linear
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, save_mem=True, use_bn=True):
        super(GCN, self).__init__()

        self.convs = nn.ModuleList()
        # self.convs.append(
        #     GCNConv(in_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=not save_mem))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            # self.convs.append(
            #     GCNConv(hidden_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=not save_mem))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # self.convs.append(
        #     GCNConv(hidden_channels, out_channels, cached=not save_mem, normalize=not save_mem))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, cached=not save_mem))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x = data.graph['node_feat']
        edge_index=data.graph['edge_index']
        edge_weight=data.graph['edge_weight'] if 'edge_weight' in data.graph else None
        for i, conv in enumerate(self.convs[:-1]):
            if edge_weight is None:
                x = conv(x, edge_index)
            else:
                x=conv(x,edge_index,edge_weight)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, data.graph['edge_index'])
        return x

class AnchorGCNLayer(nn.Module):
    """
    Simple AnchorGCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False, batch_norm=False):
        super(AnchorGCNLayer, self).__init__()
        # self.weight = torch.Tensor(in_features, out_features)
        # self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        
        self.lin = Linear(in_features, out_features, bias=False,
                          weight_initializer='glorot')
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        self.bn = nn.BatchNorm1d(out_features) if batch_norm else None

    def reset_parameters(self):
        # super().reset_parameters()
        self.lin.reset_parameters()
        # nn.init.xavier_uniform_(self.weight)
        
        if self.bn is not None:
            self.bn.reset_parameters()
        if self.bias is not None:
            zeros(self.bias)


    def forward(self, input, adj, anchor_mp=True):
        # support = torch.matmul(input, self.weight) #n*s
        support = self.lin(input)

        if anchor_mp:
            node_anchor_adj = adj
            anchor_norm = normal_diag(adj, dim=-2) #s*s
            node_norm = normal_diag(adj, dim=-1) #n*n
            # node_norm = torch.diag(1.0 / torch.clamp(torch.sparse.sum(node_anchor_adj, dim=-1).to_dense(),\
            #                                         min=VERY_SMALL_NUMBER))  
            # anchor_norm = torch.diag(1.0 / torch.clamp(torch.sparse.sum(node_anchor_adj, dim=-2).to_dense(),\
            #                                            min=VERY_SMALL_NUMBER))  

            anchor_diff = torch.sparse.mm(node_anchor_adj, anchor_norm)  
            node_diff = torch.sparse.mm(node_norm, node_anchor_adj) 

            # node_norm = node_anchor_adj / torch.clamp(torch.sum(node_anchor_adj, dim=-2, keepdim=True), min=VERY_SMALL_NUMBER)
            # anchor_norm = node_anchor_adj / torch.clamp(torch.sum(node_anchor_adj, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)
            output = torch.sparse.mm(node_diff, torch.sparse.mm(anchor_diff.transpose(-1, -2), support))

        else:
            node_adj = adj
            output = torch.sparse.mm(node_adj, support)

        if self.bias is not None:
            output = output + self.bias

        if self.bn is not None :
            output = self.compute_bn(output)

        return output

    def compute_bn(self, x):
        if len(x.shape) == 2:
            return self.bn(x)
        else:
            return self.bn(x.view(-1, x.size(-1))).view(x.size())


class AnchorGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, graph_hops, num_anchors, dropout=0.5, bias= False, batch_norm=True):
        super(AnchorGCN, self).__init__()
        self.dropout = dropout
        self.num_anchors = num_anchors
        self.graph_encoders = nn.ModuleList()
        self.graph_encoders.append(AnchorGCNLayer(nfeat, nhid, bias = bias, batch_norm=batch_norm))

        for _ in range(graph_hops - 2):
            self.graph_encoders.append(AnchorGCNLayer(nhid, nhid, bias = bias, batch_norm=batch_norm))

        self.graph_encoders.append(AnchorGCNLayer(nhid, nclass, bias = bias, batch_norm=False))

    def reset_parameters(self):
        for conv in self.graph_encoders:
            conv.reset_parameters()


    def forward(self, dataset):
        x = dataset.graph['node_feat']
        adj = dataset.graph['edge_index']
        _, anchor_idx = sample_anchors(x, self.num_anchors)
        mask = torch.isin(adj[1], anchor_idx)
        node_anchor_adj = adj[:,mask]
        # idx = torch.arange(self.num_anchors)
        node_anchor_adj[1] = torch.tensor([torch.where(anchor_idx == value)[0][0] for value in node_anchor_adj[1]])
        # node_anchor_adj[1] = torch.searchsorted(anchor_idx, node_anchor_adj[1])
        node_anchor_adj = torch.sparse_coo_tensor(
            indices = node_anchor_adj,
            values = torch.ones_like(node_anchor_adj[0]).float(),
            size = (x.size(0), self.num_anchors)
        )

        for i, encoder in enumerate(self.graph_encoders[:-1]):
            x = encoder(x, node_anchor_adj)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.graph_encoders[-1](x, node_anchor_adj)

        return x
