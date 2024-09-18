import argparse
import copy
import os
import random
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils import class_rand_splits, eval_acc, evaluate, load_fixed_splits
from dataset import load_nc_dataset
from logger import Logger
from parse import parse_method, parser_add_default_args, parser_add_main_args
from torch_geometric.utils import (add_self_loops, remove_self_loops,
                                   to_undirected)
from utils.graph_loss import GraphLoss
import time
from progress.bar import IncrementalBar
import pandas as pd
warnings.filterwarnings('ignore')

# NOTE: for consistent data splits, see data_utils.rand_train_test_idx


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def make_print(method):
    print_str = ''
    if args.rand_split_class:
        print_str += f'label per class:{args.label_num_per_class}, valid:{args.valid_num},test:{args.test_num}\n'
    if method == 'ours' or 'ours_anchor':
        print_str += f'method: {args.method} hidden: {args.hidden_channels} layers: {args.num_layers } ours_layers:{args.ours_layers} lr:{args.lr} num_anchors:{args.num_anchors} \n'
        if not args.use_graph:
            return print_str
        # if args.backbone == 'gcn':
        #     print_str += f'backbone:{args.backbone}, layers:{args.num_layers} hidden: {args.hidden_channels} lr:{args.lr} decay:{args.weight_decay} dropout:{args.dropout}\n'
    else:
        print_str += f'method: {args.method} hidden: {args.hidden_channels} lr:{args.lr}\n'
    return print_str


def run_training_process(args):

    # device
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(args.device)
                            ) if torch.cuda.is_available() else torch.device("cpu")


    ### Load and preprocess data ###
    dataset = load_nc_dataset(args)

    if len(dataset.label.shape) == 1: #only one graph
        dataset.label = dataset.label.unsqueeze(1) #shape:n*1
    dataset.label = dataset.label.to(device)

    # dataset_name = args.dataset

    # if args.rand_split:
    #     split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
    #                     for _ in range(args.runs)]
    # elif args.rand_split_class:
    #     split_idx_lst = [class_rand_splits(
    #         dataset.label, args.label_num_per_class, args.valid_num, args.test_num)]
    if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
        split_idx_lst = [dataset.load_fixed_splits()
                        for _ in range(args.runs)]       
        
    if args.dataset in ['Photo']:
        split_idx_lst = [dataset.load_fixed_splits()
                        for _ in range(args.runs)]    

    elif args.dataset in ['ogbn-proteins', 'ogbn-arxiv', 'ogbn-products']:
        split_idx_lst = [dataset.load_fixed_splits()
                        for _ in range(args.runs)]
    else:
        split_idx_lst = load_fixed_splits(
            dataset, name=args.dataset, protocol=args.protocol)

    n = dataset.graph['num_nodes']
    # infer the number of classes for non one-hot and one-hot labels
    c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    d = dataset.graph['node_feat'].shape[1]

    _shape = dataset.graph['node_feat'].shape
    print(f'features shape={_shape}')

    # whether or not to symmetrize
    if args.dataset not in {'deezer-europe'}:
        dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

    dataset.graph['edge_index'], dataset.graph['node_feat'] = \
        dataset.graph['edge_index'].to(
            device), dataset.graph['node_feat'].to(device)

    print(f"num nodes {n} | num classes {c} | num node feats {d}")

    ### Load method ###
    # model = parse_method(args.method, args, c, d, device)

    # using rocauc as the eval function
    # if args.dataset in ('deezer-europe'):
    #     criterion = nn.BCEWithLogitsLoss()
    #     # criterion = nn.CrossEntropyLoss()
    # else:
    #     criterion = nn.NLLLoss()
    #     graph_loss = GraphLoss()

    eval_func = eval_acc

    logger = Logger(args.runs, args)

    # model.train()

    ### Training loop ###
    # patience = 0
    # 优化器
    # if args.method in ['ours', 'ours_anchor']:
    #     optimizer = torch.optim.Adam([
    #         {'params': model.params1, 'weight_decay': args.ours_weight_decay}, 
    #         {'params': model.params2, 'weight_decay': args.weight_decay}
    #     ],
    #         lr=args.lr)
    #     # optimizer = torch.optim.Adam([
    #     #     {'params': model.params1}, 
    #     #     {'params': model.params2}
    #     # ],
    #     #     lr=args.lr)
    #     # param1 refers to parameters involved in attention module;
    #     # param2 refers to parameters involved in gnn module.
    # else:
    #     optimizer = torch.optim.Adam(
    #         model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
        

    # 随机训练{run}次
    for run in range(args.runs):
        if args.dataset in ['Cora', 'CiteSeer', 'PubMed'] and args.protocol == 'semi':
            split_idx = split_idx_lst[0]
        else:
            split_idx = split_idx_lst[0]
        train_idx = split_idx['train'].to(device)
        # 初始化模型
        model = parse_method(args.method, args, c, d, device)
        model.reset_parameters()
        # graph_loss.reset_avg_accuracy()
        if args.dataset in ('deezer-europe'):
            criterion = nn.BCEWithLogitsLoss()
        # criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.NLLLoss()
            graph_loss = GraphLoss()

        if args.method in ['ours', 'ours_anchor']:
            optimizer = torch.optim.Adam([
                {'params': model.params1, 'weight_decay': args.ours_weight_decay}, 
                {'params': model.params2, 'weight_decay': args.weight_decay}
            ],
                lr=args.lr)
            # optimizer = torch.optim.Adam([
            #     {'params': model.params1}, 
            #     {'params': model.params2}
            # ],
            #     lr=args.lr)
            # param1 refers to parameters involved in attention module;
            # param2 refers to parameters involved in gnn module.
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
        model.train()       

        best_val = float('-inf')
        patience = 0
        # Loss = []
        # Graph_loss = []
        for epoch in range(args.epochs):
            
            model.train()
            optimizer.zero_grad()
            
            if args.method in ['gcn' ,'AnchorGCN']:
                out = model(dataset)
            if args.method in ['ours', 'ours_anchor']:
                out, logprobs = model(dataset)

            if args.dataset in ('deezer-europe'):
                if dataset.label.shape[1] == 1:
                    true_label = F.one_hot(
                        dataset.label, dataset.label.max() + 1).squeeze(1)
                else:
                    true_label = dataset.label

                loss = criterion(out[train_idx], true_label.squeeze(1)[
                    train_idx].to(torch.float))
            else:
                out = F.log_softmax(out, dim=1)
                loss = criterion(
                    out[train_idx], dataset.label.squeeze(1)[train_idx])
                true_label = dataset.label
            loss.backward()
            # correct_t = ((out[train_idx].argmax(-1) == true_label[train_idx].argmax(-1)).float().mean().item())
            # graph loss
            if args.method in ['ours', 'ours_anchor']:
                GraphLoss_ = graph_loss(out[train_idx], true_label[train_idx], logprobs[train_idx,:])

                GraphLoss_.backward()
            

            optimizer.step()
            # Loss.append(loss.cpu().item())
            # Graph_loss.append(GraphLoss_.cpu().item())

            result = evaluate(model, dataset, split_idx,
                            eval_func, criterion, args)
            logger.add_result(run, result[:-1]) 

            if result[1] > best_val: #valid_acc
                best_val = result[1]
                patience = 0
            else:
                patience += 1
                if patience >= args.patience:
                    break
            
            if epoch % args.display_step == 0:

                print(f'Epoch: {epoch:02d}, '
                    f'Loss: {loss:.4f}, '
                    f'Train: {100 * result[0]:.2f}%, '
                    f'Valid: {100 * result[1]:.2f}%, '
                    f'Test: {100 * result[2]:.2f}%')
                progress_bar = IncrementalBar(f'Training: {epoch}', max=args.display_step)
            progress_bar.next()
        logger.print_statistics(run) 
#         data = {'Loss': Loss, 'Graph Loss': Graph_loss}
#         df = pd.DataFrame(data)

    results = logger.print_statistics() 
    print('\n')
    print(results)

    return results


if __name__ == "__main__":
        ### Parse args ###
    parser = argparse.ArgumentParser(description='General Training Pipeline')
    parser_add_main_args(parser)
    args = parser.parse_args()
    parser_add_default_args(args)
    print(args)


    fix_seed(args.seed)
    results = run_training_process(args)
    out_folder = 'results'
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    file_name = f'{args.dataset}_{args.method}'
    # if args.method == 'ours' or 'ours_anchor':
    #     file_name += '_' + args.backbone
    file_name += '.txt'
    out_path = os.path.join(out_folder, file_name)
    with open(out_path, 'a+') as f:
        print_str = make_print(args.method)
        f.write(print_str)
        f.write(results)
        f.write('\n\n')
