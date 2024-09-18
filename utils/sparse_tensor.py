import torch
from .constants import SMALL_NUMBER

def edge_index_to_sparse_tensor(edge_index, num_nodes_1, num_nodes_2):
    values = torch.ones_like(edge_index[0]).float()  
    sparse_tensor = torch.sparse_coo_tensor(
        indices=edge_index,  
        values=values,  
        size=(num_nodes_1, num_nodes_2)  
    )
    return sparse_tensor

def normal_diag(adj, dim=-1):
    count = torch.sparse.sum(adj, dim=dim).to_dense()  
    diag_indices = torch.arange(count.size(0)).unsqueeze(0)
    inv_row_sum = 1.0 / torch.clamp(count, min=SMALL_NUMBER)  
    diag_matrix = torch.sparse_coo_tensor(
        indices=torch.concat([diag_indices, diag_indices],dim=-2).to(count.device),  
        values=inv_row_sum,  
        size=(count.size(0), count.size(0))  
    )
    return diag_matrix