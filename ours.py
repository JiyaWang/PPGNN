import math
import os
from multiprocessing.sharedctypes import Value
from torch_geometric.utils import to_dense_adj
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import GCN, AnchorGCN, AnchorGCNLayer
from torch_geometric.utils import degree
from torch_sparse import SparseTensor, matmul
from torch.nn.parameter import Parameter
from torch_geometric.utils import to_dense_adj, add_self_loops
from utils.distance import *
from utils.anchor import anchor_aggregation, sample_anchors
from utils.sparse_tensor import *
from utils.constants import VERY_SMALL_NUMBER, SUPER_SMALL_NUMBER
from torch_geometric.nn import GCNConv

def gumbel_top_k(logits, k):
    n, s = logits.shape
    q = torch.rand_like(logits) + VERY_SMALL_NUMBER  # dim: n*s
    lq = (logits - torch.log(-torch.log(q)))  # dim: n*s

    lprobs, indices = torch.topk(-lq, k)  # dim:n*k
    values_1 = torch.ones_like(lprobs)
    # rows = torch.arange(n).view(1, n, 1).to(logits.device).repeat(b, 1, self.k)
    # edges = torch.stack((indices.view(b, -1), rows.view(b, -1)), -2)
    edges = (0 * torch.ones_like(logits)).scatter_(-1, indices, values_1).to(logits.device)
    # edges = torch.sparse_coo_tensor(
    #     indices = indices,
    #     values = values_1,
    #     size = (n , s)
    # )

        # return (edges + (torch.arange(b).to(logits.device) * n)[:, None, None]).transpose(0, 1).reshape(2,
        #                                                                                                 -1), lprobs
    return edges, lprobs


def probability_passing(logits, graph_init):
    n, _= logits.shape

    log_prob=logits.squeeze()

    prob= (-log_prob).exp()
    graph_init , _ = add_self_loops(graph_init) 
    graph_init = edge_index_to_sparse_tensor(graph_init.detach(),n,n) 
    # graph_init = to_dense_adj(graph_init.detach()).squeeze()

    # count = torch.sparse.sum(graph_init,dim=-1).to_dense() 
    # normal_D = torch.diag(1.0 / torch.clamp(torch.sparse.sum(graph_init, dim=-1).to_dense(),\
    #                                             min=1e-8))
    # diagonal_tensor = torch.eye(graph_init.size(0), graph_init.size(1)).to_sparse()
    # diagonal_tensor = torch.sparse_coo_tensor(torch.nonzero(diagonal_tensor), diagonal_tensor[torch.nonzero(diagonal_tensor)])
    # graph_init = torch.sparse.add(graph_init, diagonal_tensor)
    normal_D = normal_diag(graph_init) 
    # graph_init = graph_init.to_sparse()
    merge_matrix = torch.sparse.mm(torch.sparse.mm(normal_D, graph_init),prob).clamp_(min = SUPER_SMALL_NUMBER).log()
    # merge_matrix_1 = (torch.sparse.mm(graph_init,prob).clamp_(min = 1e-24)).log()-(count.clamp_(min=1e-8)).log()
    # merge_matrix_2 = torch.transpose(merge_matrix_1,dim0=-1,dim1=-2)
    
    # merge_matrix=  torch.logsumexp(torch.cat([merge_matrix_1.unsqueeze(-1), merge_matrix_2.unsqueeze(-1)], dim=-1),dim=-1) - math.log(2.)

    return -merge_matrix #b*n*s

def prob_boolean_product_sparse(logprobs, graph_init):
    """
    Boolean product between probability graph and the original graph (sparse version).

    :param logprobs: log probability of feature graph. 
    :param A_init: the original graph structure.
    :return: a merge graph.
    """


    logprobs=logprobs.squeeze()

    prob= (-logprobs).exp()
    graph_init , _ = add_self_loops(graph_init)
    
    graph_init = to_dense_adj(graph_init.detach()).squeeze()

    count = torch.sum(graph_init,dim=-1,keepdim=True) 

    graph_init = graph_init.to_sparse()

    merge_matrix_1 = (torch.sparse.mm(graph_init,prob).clamp_(min = 1e-24)).log()-(count.clamp_(min=1e-8)).log()
    merge_matrix_2 = torch.transpose(merge_matrix_1,dim0=-1,dim1=-2)
    
    merge_matrix=  torch.logsumexp(torch.cat([merge_matrix_1.unsqueeze(-1), merge_matrix_2.unsqueeze(-1)], dim=-1),dim=-1) - math.log(2.)

    return -merge_matrix

class similarity_layer(nn.Module):
    def __init__(self, embed_f, num_anchor, method, k=5, use_graph = True, use_anchor = True):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(0. ).float())
        self.use_graph = use_graph
        self.k = k
        self.embed_f = embed_f
        self.num_anchor = num_anchor
        self.method = method
        self.distance = anchor_node_distances
        self.use_anchor = use_anchor

    def reset_parameters(self):
        self.embed_f.reset_parameters()
        self.temperature.data.fill_(0.0) 

    def forward(self, x, edges, A_init):
        # x = data.graph['node_feat']
        # edge_index = data.graph['edge_index']
        # edge_weight = data.graph['edge_weight'] if 'edge_weight' in data.graph else None
        
        # x = self.embed_f(x, edges) #gnn
        if self.method == 'linear':
            x = self.embed_f(x) #linear x:n*d_hidden
        else:
            x = self.embed_f(x, edges)
        
        if self.use_anchor:
            anchor_vec, _ = sample_anchors(x, self.num_anchor) #tensor:s*d_hidden
            D = self.distance(x, anchor_vec) #n*s （node-anchor distance matrix）
        else:
            D = self.distance(x, x)

        if self.training:
            
            # log_prob = D * torch.exp(torch.clamp(self.temperature, -5, 5)) #log probability(n*s)
            log_prob = D * torch.exp(torch.clamp(self.temperature, -5, 5))
            if self.use_graph:
                logprobs = probability_passing(log_prob, A_init) #construct graph structure
                # logprobs = prob_boolean_product_sparse(logprobs, A_init)
            else:
                logprobs = log_prob
            node_anchor_edges_hat, logprobs = gumbel_top_k(logprobs, self.k)

            node_anchor_edges_hat = node_anchor_edges_hat.to_sparse()

        else:
            with torch.no_grad():
                
                log_prob = D * torch.exp(torch.clamp(self.temperature, -5, 5))

                if self.use_graph:
                    logprobs = probability_passing(log_prob, A_init) 
                else:
                    logprobs = log_prob
                    # logprobs = prob_boolean_product_sparse(logprobs, A_init)
                node_anchor_edges_hat, logprobs = gumbel_top_k(logprobs, self.k)

                node_anchor_edges_hat = node_anchor_edges_hat.to_sparse()


        return x, node_anchor_edges_hat, logprobs

# class rebuild(nn.module):
#     def __init__(self):
#         super(rebuild, self).__init__()
#         self.convs = nn.ModuleList()


class our_model_anchor(nn.Module):
    """
    Graph residual connection block based on Boolean Product.
    This class includes generating probabilty graph,computing Boolean product between probability graph and the original graph and sampling based on gumble-top-k.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_anchor, our_num_layers=1, backbone_num_layers=2, backbone_bias=True, backbone_batch_norm =True,
                 k=5, dropout=0.5, our_dropout=0.5, use_graph=True, embed_f = 'linear'):
        super(our_model_anchor, self).__init__()
        self.num_anchor = num_anchor
        self.dropout = dropout
        self.our_dropout = our_dropout
        self.method = embed_f

        self.encoder = nn.ModuleList()
        self.encoder.append(AnchorGCNLayer(in_channels, hidden_channels, bias = backbone_bias, batch_norm=backbone_batch_norm))
        for _ in range(backbone_num_layers-2):
            self.encoder.append(AnchorGCNLayer(hidden_channels, hidden_channels, bias = backbone_bias, batch_norm=backbone_batch_norm))
        self.encoder.append(AnchorGCNLayer(hidden_channels, hidden_channels, bias = backbone_bias, batch_norm=True))

        self.graph_rebuild = nn.ModuleList()
        if embed_f == 'linear':
            f_graph = nn.Linear(in_channels, hidden_channels) #in_channels:d_feature
        else:
            f_graph = GCNConv(in_channels, hidden_channels)
        self.graph_rebuild.append(similarity_layer(f_graph, num_anchor, method = self.method, k=k, use_graph=use_graph))
        
        for _ in range(our_num_layers-1):
            if embed_f == 'linear':
                f_graph = nn.Linear(hidden_channels*2, hidden_channels) #in_channels:d_feature
            else:
                f_graph = GCNConv(hidden_channels*2, hidden_channels)
            self.graph_rebuild.append(similarity_layer(f_graph, num_anchor, method = self.method, k=k, use_graph=use_graph))


        self.fc = nn.Linear(hidden_channels, out_channels)

        self.params1 = list(self.graph_rebuild.parameters())
        self.params2 = list(self.encoder.parameters()) if self.encoder is not None else []
        self.params2.extend(list(self.fc.parameters()))

    def reset_parameters(self):
        for layer in self.graph_rebuild:
            layer.reset_parameters()
        for layer in self.encoder:
            layer.reset_parameters()
        self.fc.reset_parameters()

    def forward(self, data):
        A_init = data.graph['edge_index'].detach() #initial edges
        lprobslist = [] #log probability list
        graph_x = data.graph['node_feat'].detach() #features edges
        x = data.graph['node_feat']
        edges = data.graph['edge_index']
        for i, conv in enumerate(self.encoder[:-1]):
            if i < len(self.graph_rebuild):
                graph_x, edges, lprobs = self.graph_rebuild[i](graph_x, edges, A_init)
                lprobslist.append(lprobs)
            x = conv(x, edges)
            
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training = self.training)
            if i < len(self.graph_rebuild) -1:
                graph_x = torch.dropout(F.relu(graph_x), self.our_dropout, train = self.training)
                graph_x = torch.cat([graph_x,x.detach()],-1)
                edges = edges.indices()

         #construct graph
        if len(self.graph_rebuild) == len(self.encoder):
            graph_x, edges, lprobs = self.graph_rebuild[-1](graph_x, edges, A_init)
            lprobslist.append(lprobs)

        # else:
        #     edges = edge_index_to_sparse_tensor(edges, x.size(0), self.num_anchor)
        x = self.encoder[-1](x, edges)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training = self.training)
        return self.fc(x), torch.stack(lprobslist, -1) if len(lprobslist) > 0 else None
 
class our_model(nn.Module):
    """
    Graph residual connection block based on Boolean Product.
    This class includes generating probabilty graph,computing probability Passing between probability graph and the original graph and sampling based on gumble-top-k.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_anchor, our_num_layers=1, backbone_num_layers=2, backbone_bias=True, backbone_batch_norm =True,
                 k=5, dropout=0.5, our_dropout=0.5, use_graph=True, embed_f = 'linear'):
        super(our_model, self).__init__()
        self.num_anchor = num_anchor
        self.dropout = dropout
        self.our_dropout = our_dropout
        self.method = embed_f
        self.encoder = nn.ModuleList()
        self.encoder.append(GCNConv(in_channels, hidden_channels))
        for _ in range(backbone_num_layers-2):
            self.encoder.append(GCNConv(hidden_channels, hidden_channels))
        self.encoder.append(GCNConv(hidden_channels, hidden_channels))

        self.graph_rebuild = nn.ModuleList()
        if embed_f == 'linear':
            f_graph = nn.Linear(in_channels, hidden_channels) #in_channels:d_feature
        else:
            f_graph = GCNConv(in_channels, hidden_channels)
        self.graph_rebuild.append(similarity_layer(f_graph, num_anchor, method = self.method, k=k, use_graph=use_graph, use_anchor = False))
        for _ in range(our_num_layers-1):
            if embed_f == 'linear':
                f_graph = nn.Linear(hidden_channels*2, hidden_channels) #in_channels:d_feature
            else:
                f_graph = GCNConv(hidden_channels*2, hidden_channels)
            self.graph_rebuild.append(similarity_layer(f_graph, num_anchor, method = self.method, k=k, use_graph=use_graph, use_anchor = False))
        self.fc = nn.Linear(hidden_channels, out_channels)

        self.params1 = list(self.graph_rebuild.parameters())
        self.params2 = list(self.encoder.parameters()) if self.encoder is not None else []
        self.params2.extend(list(self.fc.parameters()))

    def reset_parameters(self):
        for layer in self.graph_rebuild:
            layer.reset_parameters()
        for layer in self.encoder:
            layer.reset_parameters()
        self.fc.reset_parameters()

    def forward(self, data):
        A_init = data.graph['edge_index'].detach() #initial edges
        lprobslist = [] #log probability
        graph_x = data.graph['node_feat'].detach() #features edges
        x = data.graph['node_feat']
        edges = data.graph['edge_index']
        for i, conv in enumerate(self.encoder[:-1]):
            if i < len(self.graph_rebuild):
                graph_x, edges, lprobs = self.graph_rebuild[i](graph_x, edges, A_init)
                lprobslist.append(lprobs)
            x = conv(x, edges.indices())

            x = F.relu(x)
            x = F.dropout(x, self.dropout, training = self.training)
            if i < len(self.graph_rebuild):
                graph_x = torch.dropout(F.relu(graph_x), self.our_dropout, train = self.training)
                graph_x = torch.cat([graph_x,x.detach()],-1)

         #construct graph
        if len(self.graph_rebuild) == len(self.encoder):
            graph_x, edges, lprobs = self.graph_rebuild[-1](graph_x, edges, A_init)
            lprobslist.append(lprobs)
        # else:
        #     edges = edge_index_to_sparse_tensor(edges, x.size(0), self.num_anchor)
        x = self.encoder[-1](x, edges.indices())
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training = self.training)
        return self.fc(x), torch.stack(lprobslist, -1) if len(lprobslist) > 0 else None
#  # MLP module
# class MLP(nn.Module): 
#     def __init__(self, layers_size,final_activation=False, dropout=0):
#         super(MLP, self).__init__()
#         layers = []
#         for li in range(1,len(layers_size)):
#             if dropout>0:
#                 layers.append(nn.Dropout(dropout))
#             layers.append(nn.Linear(layers_size[li-1],layers_size[li]))
#             if li==len(layers_size)-1 and not final_activation:
#                 continue
#             layers.append(nn.LeakyReLU(0.1))
            
            
#         self.MLP = nn.Sequential(*layers)
        
#     def forward(self, x, e=None):
#         x = self.MLP(x)
#         return x
    
# # identity mapping
# class Identity(nn.Module):
#     def __init__(self,retparam=None):
#         self.retparam=retparam
#         super(Identity, self).__init__()
        
#     def forward(self, *params):
#         if self.retparam is not None:
#             return params[self.retparam]
#         return params