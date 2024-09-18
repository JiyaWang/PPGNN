import torch
import torch.nn as nn
from .constants import VERY_SMALL_NUMBER

def sample_anchors(node_vec, s):
    idx = torch.randperm(node_vec.size(-2))[:s].to(node_vec.device)
    sampled_node_vec = node_vec.index_select(dim=-2, index=idx)
    return sampled_node_vec, idx

class anchor_aggregation(nn.Module):

    '''compute anchor matrix by aggregation with learnable parameters
    anchor_num: number of anchors;
    n_number: number of nodes.
    output:(s*n)features matrix for anchors.'''
    
    def __init__(self, anchor_num, n_number):
        super(anchor_aggregation, self).__init__()
        self.weight = torch.Tensor(anchor_num, n_number)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        self.relu = nn.LeakyReLU(0.1)


    def forward(self, input):
        output = torch.matmul(self.weight, input) #n*s
        return self.relu(output)



# def batch_sample_anchors(node_vec, ratio, node_mask=None, device=None):
#     idx = []
#     num_anchors = []
#     max_num_anchors = 0
#     for i in range(node_vec.size(0)):
#         tmp_num_nodes = int(node_mask[i].sum().item())
#         tmp_num_anchors = int(ratio * tmp_num_nodes)
#         g_idx = torch.randperm(tmp_num_nodes)[:tmp_num_anchors]
#         idx.append(g_idx)
#         num_anchors.append(len(g_idx))

#         if max_num_anchors < len(g_idx):
#             max_num_anchors = len(g_idx)

#     anchor_vec = batch_select_from_tensor(node_vec, idx, max_num_anchors, device)
#     anchor_mask = create_mask(num_anchors, max_num_anchors, device)

#     return anchor_vec, anchor_mask, idx, max_num_anchors

# def batch_select_from_tensor(node_vec, idx, max_num_anchors, device=None):
#     anchor_vec = []
#     for i in range(node_vec.size(0)):
#         tmp_anchor_vec = node_vec[i][idx[i]]
#         if len(tmp_anchor_vec) < max_num_anchors:
#             dummy_anchor_vec = to_cuda(torch.zeros((max_num_anchors - len(tmp_anchor_vec), node_vec.size(-1))), device)
#             tmp_anchor_vec = torch.cat([tmp_anchor_vec, dummy_anchor_vec], dim=-2)
#         anchor_vec.append(tmp_anchor_vec)

#     anchor_vec = torch.stack(anchor_vec, 0)

    # return anchor_vec

def compute_anchor_adj(node_anchor_adj, anchor_mask=None):
    '''Can be more memory-efficient'''
    anchor_node_adj = node_anchor_adj.transpose(-1, -2) 
    anchor_norm = torch.clamp(anchor_node_adj.sum(dim=-2), min=VERY_SMALL_NUMBER) ** -1 # Delta #n
    # anchor_adj = torch.matmul(anchor_node_adj, torch.matmul(torch.diag(anchor_norm), node_anchor_adj))
    anchor_adj = torch.matmul(anchor_node_adj, anchor_norm.unsqueeze(-1) * node_anchor_adj) 

    markoff_value = 0
    if anchor_mask is not None:
        anchor_adj = anchor_adj.masked_fill_(1 - anchor_mask.byte().unsqueeze(-1), markoff_value)
        anchor_adj = anchor_adj.masked_fill_(1 - anchor_mask.byte().unsqueeze(-2), markoff_value)

    return anchor_adj

def compute_node_adj(node_anchor_adj, anchor_mask=None):

    anchor_node_adj = node_anchor_adj.transpose(-1, -2) 
    anchor_norm = torch.clamp(anchor_node_adj.sum(dim=-2), min=VERY_SMALL_NUMBER) ** -1 # Delta
    # anchor_adj = torch.matmul(anchor_node_adj, torch.matmul(torch.diag(anchor_norm), node_anchor_adj))
    anchor_adj = torch.matmul(anchor_node_adj, anchor_norm.unsqueeze(-1) * node_anchor_adj) 

    markoff_value = 0
    if anchor_mask is not None:
        anchor_adj = anchor_adj.masked_fill_(1 - anchor_mask.byte().unsqueeze(-1), markoff_value)
        anchor_adj = anchor_adj.masked_fill_(1 - anchor_mask.byte().unsqueeze(-2), markoff_value)

    return anchor_adj
