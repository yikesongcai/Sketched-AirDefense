import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    
    # federated arguments
    parser.add_argument('--fed', type=str, default='fedavg', help="federated optimization algorithm")
    parser.add_argument('--mu', type=float, default=1e-2, help='hyper parameter for fedprox')
    parser.add_argument('--rounds', type=int, default=800, help="total number of communication rounds")
    parser.add_argument('--num_users', type=int, default=41, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1, help="fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1, help="number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=50, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.005, help="client learning rate")
    parser.add_argument('--momentum', type=float, default=0, help="SGD momentum (default: 0.5)")
    parser.add_argument('--classwise', type=int, default=10, help="number of images for each class (global dataset)")
    parser.add_argument('--alpha', type=float, default=0.1, help="Dirichlet concentration parameter for non-IID partitioning (closer to 0 is more non-IID)")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--sampling', type=str, default='noniid', help="sampling method")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of images")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=2, help='random seed (default: 1)')
    parser.add_argument('--tsboard', action='store_true', help='tensorboard')
    parser.add_argument('--debug', action='store_true', help='debug')
    parser.add_argument('--opt', type=str, default='opted', help="status of optimization")
    parser.add_argument('--trans', type=str, default='proposed', help="transmission")
    
    
    # wireless arguments
    parser.add_argument('--gth', type=float, default=0.1, help="truncation threshold")
    parser.add_argument('--power', type=float, default=-6, help="maximum transmit power")
    
    # Byzatine arguments
    parser.add_argument('--num_byz', type=int, default=6, help="number of Byzantine")
    parser.add_argument('--num_cluster', type=int, default=8, help="number of clusters")
    parser.add_argument('--optw', type=str, default='opt', help="weight adaptive")
    parser.add_argument('--clustering', type=str, default='location', help="clustering")
    parser.add_argument('--attack', type=str, default='omni',
                        help="attack type: omni, gaussian, label, null_space, slow_poison, predictor_proxy")
    parser.add_argument('--V', type=float, default=1e3, help="Lyapunov")

    # Adaptive attack parameters (for null_space, slow_poison, predictor_proxy)
    parser.add_argument('--attack_strength', type=float, default=1.0,
                        help="attack strength for null_space attack (gamma)")
    parser.add_argument('--poison_alpha', type=float, default=0.05,
                        help="interpolation factor for slow_poison attack")
    parser.add_argument('--poison_decay', type=float, default=0.99,
                        help="decay rate for slow_poison alpha over time")
    parser.add_argument('--proxy_hidden', type=int, default=64,
                        help="hidden dimension for predictor_proxy local LSTM")
    parser.add_argument('--threshold_margin', type=float, default=0.1,
                        help="safety margin for predictor_proxy attack (delta)")

    # Sketched-AirDefense arguments
    parser.add_argument('--sketch_dim', type=int, default=128, help="dimension of gradient sketch")
    parser.add_argument('--pred_hidden', type=int, default=64, help="hidden dimension of trajectory predictor")
    parser.add_argument('--window_size', type=int, default=5, help="window size for trajectory prediction")
    parser.add_argument('--defense_method', type=str, default='sketched', help="defense method: sketched, none")
    parser.add_argument('--pred_lr', type=float, default=0.001, help="learning rate for trajectory predictor")
    parser.add_argument('--anomaly_threshold', type=float, default=0.2, help="top percentage of clusters to filter (0-1)")

    # Defense-aware predictor arguments (new)
    parser.add_argument('--lambda_adv', type=float, default=0.5, help="weight for adversarial loss in defense-aware training")
    parser.add_argument('--warmup_rounds', type=int, default=3, help="soft-landing warmup rounds after cluster switch")
    parser.add_argument('--use_defense_aware', action='store_true', default=False, help="use defense-aware LSTM predictor instead of GRU")
    parser.add_argument('--no_weight_pred', action='store_true', default=False, help="skip weight prediction and use uniform constant weights")


    args = parser.parse_args()
    
    return args
