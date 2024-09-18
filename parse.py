from models import *
from ours import *
'''
This document contains the final model and the paramters we can adjust. 
The function parse_method combines the gcn structure and the our method.
'''

def parse_method(method, args, c, d, device):
    # only use the gcn
    if method == 'gcn':
        model = GCN(in_channels=d,
                    hidden_channels=args.hidden_channels,
                    out_channels=c,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    use_bn=args.use_bn).to(device)
        # the parameters in gcn contains the dimensions of input and output, the number of layers（main!!）
    # use the gcn and attention.

    elif method == 'AnchorGCN':
        model = AnchorGCN(nfeat = d, 
                          nhid = args.hidden_channels, 
                          nclass = c, 
                          graph_hops = args.num_layers, 
                          num_anchors= args.num_anchors,
                          dropout=args.dropout, 
                          batch_norm=args.use_bn).to(device)

    elif method == 'ours_anchor':
        model = our_model_anchor(in_channels = d, 
                              hidden_channels = args.hidden_channels, 
                              out_channels = c, 
                              num_anchor = args.num_anchors,
                              our_num_layers=args.ours_layers, 
                              backbone_num_layers = args.num_layers,
                              k=args.k, 
                              dropout=args.dropout,
                              our_dropout = args.ours_dropout,
                              use_graph = args.use_graph,
                              embed_f = 'gcn'
                              ).to(device)
    elif method == 'ours':
        model = our_model(in_channels = d, 
                              hidden_channels = args.hidden_channels, 
                              out_channels = c, 
                              num_anchor = args.num_anchors,
                              our_num_layers=args.ours_layers, 
                              backbone_num_layers = args.num_layers,
                              k=args.k, 
                              dropout=args.dropout,
                              our_dropout = args.ours_dropout,
                              use_graph = args.use_graph,
                              embed_f = 'gcn'
                              ).to(device)
                    #  use_bn=args.use_bn, use_residual=args.ours_use_residual, use_graph=args.use_graph, 
                    #  use_weight=args.ours_use_weight, use_act=args.ours_use_act, graph_weight=args.graph_weight, gnn=backbone, 
                    #  aggregate=args.aggregate).to(device)
        # else:
        #     model = our_model(d, args.hidden_channels, c, num_layers=args.num_layers, alpha=args.alpha, 
        #                  dropout=args.dropout, num_heads=args.num_heads,
        #              use_bn=args.use_bn, use_residual=args.ours_use_residual, use_graph=args.use_graph, 
        #              use_weight=args.ours_use_weight, use_act=args.ours_use_act, graph_weight=args.graph_weight, 
        #              aggregate=args.aggregate).to(device)
    else:
        raise ValueError(f'Invalid method {method}')
    return model


def parser_add_main_args(parser):
    # dataset and evaluation
    # parser.add_argument('--data_dir', type=str, default='../../../NodeFormer/data/')
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--cpu', action='store_true')

    # training process
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--runs', type=int, default=5,
                        help='number of distinct runs')
    parser.add_argument('--patience', type=int, default=200,
                        help='early stopping patience.')
    parser.add_argument('--display_step', type=int,
                        default=50, help='how often to print')   
    
    # dataset
    parser.add_argument('--train_prop', type=float, default=.5,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.25,
                        help='validation label proportion')
    parser.add_argument('--protocol', type=str, default='semi',
                        help='protocol for cora datasets, semi or supervised')
    
    parser.add_argument('--rand_split', default = False,
                        help='use random splits')
    parser.add_argument('--rand_split_class', default = True,
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=1000,
                        help='Total number of test')
    parser.add_argument('--no_feat_norm', action='store_true', default=True,
                        help='Not use feature normalization.')
    # model_backbone
    parser.add_argument('--method', type=str, default='ours_anchor')
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers for deep methods')

    parser.add_argument('--use_bn', default = True, action='store_true', help='use layernorm')
    # parser.add_argument('--use_residual', action='store_true', default = True,
    #                     help='use residual link for each GNN layer')

    #model_ours
    parser.add_argument('--use_graph', default = True, help='use pos emb')
    parser.add_argument('--embed_f', type=str, default = 'gcn')
    parser.add_argument('--num_anchors', type=int, default=1000)
    parser.add_argument('--ours_layers', type=int, default=1,
                    help='gnn layer.')
    parser.add_argument('--k',type = int, default=3)
    # parser.add_argument('--use_weight', action='store_true',
    #                     help='use weight for GNN convolution')
    # parser.add_argument('--use_init', action='store_true', help='use initial feat for each GNN layer')
    # parser.add_argument('--use_act', action='store_true', help='use activation for each GNN layer')
    
    # training_gnn
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--dropout', type=float, default=0.5)

    # training_ours
    parser.add_argument('--ours_weight_decay', type=float, default = 0.,
                         help='Ours\' weight decay. Default to weight_decay.')
    parser.add_argument('--ours_dropout', type=float, default=0.5,
                        help='gnn dropout.')

    parser.add_argument('--test_eval', type=int, default = 10)
    # parser.add_argument('--ours_use_weight', action='store_true', help='use weight for trans convolution')
    # parser.add_argument('--ours_use_residual', action='store_true', help='use residual link for each trans layer')
    # parser.add_argument('--ours_use_act', action='store_true', help='use activation for each trans layer')



def parser_add_default_args(args):
    if args.method=='ours':
        if args.ours_weight_decay is None:
            args.ours_weight_decay=args.weight_decay
        if args.ours_dropout is None:
            args.ours_dropout=args.dropout