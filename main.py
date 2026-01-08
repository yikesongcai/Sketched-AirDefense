import torch
from torchvision import datasets, transforms
# from torch.utils.tensorboard import SummaryWriter

import copy
import numpy as np
import random
from tqdm import trange
import matplotlib.pyplot as plt
from collections import deque

from utils.distribute import uniform_distribute, train_dg_split
from utils.sampling import iid, noniid
from utils.options import args_parser
from src.update import ModelUpdate
from src.nets import MLP, CNN_v1, CNN_v2, ResNet18_CIFAR
from src.strategy import FedAvg, FedAvg_Byzantine, Secure_aggregation
from src.test import test_img, test_ind_img
from src.generate import generate_clients
from src.defense import (
    SketchedAirDefense, get_model_flattened_dim, Sketched_Defense_Aggregation,
    GradientSketcher, flatten_model_updates,
)
from src.client import Client, MaliciousClient

# writer = SummaryWriter()

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)     
    torch.cuda.manual_seed_all(args.seed) 
    result_acc=[]
    result_loss=[]
    
    non_iid_p=[[] for i in range(args.num_users)]
    for i in range(args.num_users-1):
        temp=np.random.randint(50,60, size=args.num_classes)
        #temp=np.zeros(args.num_classes)
        ii=i%10
        temp[ii]=1000
        temp=temp/sum(temp)
        non_iid_p[i].extend(temp)
    temp=np.ones(args.num_classes)/args.num_classes
    non_iid_p[args.num_users-1].extend(temp)

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset = datasets.MNIST('E:/programming/trail for paper/data/mnist', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('E:/programming/trail for paper/data/mnist', train=False, download=True, transform=trans_mnist)

        dg = copy.deepcopy(dataset)
        dataset_train = copy.deepcopy(dataset)
        
        dg_idx, dataset_train_idx = train_dg_split(dataset, args)
        
        dg.data, dataset_train.data = dataset.data[dg_idx], dataset.data[dataset_train_idx]
        dg.targets, dataset_train.targets = dataset.targets[dg_idx], dataset.targets[dataset_train_idx]
        
        # sample users
        if args.sampling == 'iid':
            alph, dict_users = iid(dataset_train, args.num_users)
            sumdatasize = len(dataset_train)
        elif args.sampling == 'noniid':
            alph, dict_users, sumdatasize = noniid(dataset_train, args, non_iid_p)
        else:
            exit('Error: unrecognized sampling')
    
    elif args.dataset == 'cifar':
        args.num_channels = 3
        args.num_classes = 10
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = datasets.CIFAR10('E:/programming/trail for paper/data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('E:/programming/trail for paper/data/cifar', train=False, download=True, transform=trans_cifar)
        
        dg = copy.deepcopy(dataset)
        dataset_train = copy.deepcopy(dataset)
    
        dg_idx, dataset_train_idx = train_dg_split(dataset, args)
        
        dg.targets.clear()
        dataset_train.targets.clear()

        
        dg.data, dataset_train.data = dataset.data[dg_idx], dataset.data[dataset_train_idx]
        
        for i in list(dg_idx):
            dg.targets.append(dataset[i][1])
        for i in list(dataset_train_idx):
            dataset_train.targets.append(dataset[i][1])

        # sample users
        if args.sampling == 'iid':
            alph, dict_users = iid(dataset_train, args.num_users)
            sumdatasize = len(dataset_train)
        elif args.sampling == 'noniid':
            alph, dict_users, sumdatasize = noniid(dataset_train, args, non_iid_p)
        else:
            exit('Error: unrecognized sampling')
    
    else:
        exit('Error: unrecognized dataset')
    
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        print("now use the cnn")
        net_glob = CNN_v2(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNN_v1(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    elif args.model == 'resnet18' and args.dataset == 'cifar':
        print("now use the resnet18")
        net_glob = ResNet18_CIFAR(args=args).to(args.device)
    else:
        exit('Error: unrecognized model')
    
    print(net_glob)
    net_glob.train()

    # Initialize defense / attack utilities
    defense_module = None
    attack_state = {}
    model_dim = get_model_flattened_dim(net_glob)
    sketcher = None
    history_buffer = None

    if args.defense_method == 'sketched':
        print(f"[Sketched-AirDefense] Model dimension: {model_dim}")
        print(f"[Sketched-AirDefense] Sketch dimension: {args.sketch_dim}")
        print(f"[Sketched-AirDefense] Window size: {args.window_size}")
        print(f"[Sketched-AirDefense] Number of clusters: {args.num_cluster}")
        print(f"[Sketched-AirDefense] Clustering method: {args.clustering}")
        defense_module = SketchedAirDefense(args, model_dim, args.device)
        # For sketched defense, the defense module contains its own sketcher
        sketcher = defense_module.sketcher
        history_buffer = defense_module.history_buffer 
        # Note: SketchedAirDefense uses per-cluster history, but history_buffer here is used by malicious clients
        # to observe the global state. 
        # If we use Sketched defense, the 'state' observed by attackers might be the aggregated sketches.
        
    else:
        # 为 byzantine / proposed 提供自适应攻击所需的草图器与历史缓存
        sketcher = GradientSketcher(model_dim, args.sketch_dim, args.device)
        history_buffer = deque(maxlen=args.window_size)

    # copy weights
    w_glob = net_glob.state_dict()
        
    # initialization
    initialization_stage = ModelUpdate(args=args, dataset=dataset, idxs=set(dg_idx), rnds=0, data_size = 20, flag_byz = 1)
    w_glob, _ = initialization_stage.train(local_net = copy.deepcopy(net_glob).to(args.device), net = copy.deepcopy(net_glob).to(args.device))
    net_glob.load_state_dict(w_glob)

    # Generate Channel Conditions
    distance, P, H, G = generate_clients(args)
    
    # Pass distance to defense module for location-based clustering
    if defense_module is not None:
        defense_module.set_user_distances(distance)
    
    # Initialize Clients
    clients = []
    idxs_users = np.array(range(args.num_users)) # 0 to num_users-1
    
    for idx in idxs_users:
        data_size = int(sumdatasize * alph[idx])
        
        # Check if malicious
        if idx <= args.num_byz - 1: # Users 0 to num_byz-1 are attackers
            clients.append(MaliciousClient(
                args=args,
                dataset=dataset,
                user_idx=idx,
                dict_users=dict_users,
                data_size=data_size,
                p_val=P[idx],
                channel_gain=G[:, :], # Pass full G matrix for simplicity, or we can index later
                attack_state=attack_state,
                sketcher=sketcher,
                history_buffer=history_buffer
            ))
        else:
            clients.append(Client(
                args=args,
                dataset=dataset,
                user_idx=idx,
                dict_users=dict_users,
                data_size=data_size,
                p_val=P[idx],
                channel_gain=G[:, :]
            ))
            
    num=0
    wt = net_glob.state_dict()
    reputation = np.zeros(args.num_users-1)
    q = np.zeros(args.num_users-1)
    alpha = np.ones(args.num_users-1)/(args.num_users-1)
    
    for iter in trange(args.rounds):

        num += 1
        prev_global = copy.deepcopy(wt)
        
        # Collect updates
        w_locals = []
        for client in clients:
            w_local, loss = client.train(net_glob, num, w_glob)
            w_locals.append(w_local)
            
        # update global weights
        defense_info = None
        if args.defense_method == 'sketched' and defense_module is not None:
            # Use Sketched-AirDefense with physical layer parameters
            w_glob, reputation, q, defense_info = Sketched_Defense_Aggregation(
                w_locals, args, wt, defense_module, num-1,
                P, G, H, reputation, q
            )
        elif args.trans == 'byzantine':
            w_glob = FedAvg_Byzantine(w_locals, args, P, wt, G, num-1, H)
        elif args.trans == 'proposed':
            w_glob, reputation, q = Secure_aggregation(w_locals, args, idxs_users, distance, P, wt, reputation, G, num-1, H, q)
        else:
            w_glob = FedAvg(w_locals, args)

        # 记录聚合后的全局更新供 predictor_proxy 攻击使用 (Update Global History)
        # Note: If SketchedDefense is used, it updates its history internally inside Sketched_Defense_Aggregation.
        # But if standard defense is used, we need to update history_buffer manually for the attackers to see.
        
        if args.defense_method != 'sketched' and sketcher is not None and history_buffer is not None:
             global_update_flat = flatten_model_updates(
                {k: w_glob[k] - prev_global[k] for k in sorted(w_glob.keys())},
                args.device
             )
             history_buffer.append(sketcher.sketch(global_update_flat))

        wt = copy.deepcopy(w_glob)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        acc_test, loss_test = test_img(net_glob, dataset_test, args)

        acc_ind = []
        # Testing individual accuracy (optional, can be slow)
        # for idx in idxs_users:
        #     tmp = test_ind_img(net_glob, dataset, args, set(list(dict_users[idx])))
        #     acc_ind.append(float(tmp))

        if args.debug:
            print(f"Round: {iter}")
            print(f"Test accuracy: {acc_test}")
            print(f"Test loss: {loss_test}")
            if defense_info is not None:
                print(f"[Defense] Predictor loss: {defense_info['predictor_loss']:.6f}")
                print(f"[Defense] History size: {defense_info['history_size']}")
                print(f"[Defense] Trust weights: {defense_info['trust_weights']}")
            result_acc.append(float(acc_test))
            result_loss.append(float(loss_test))
        
        # tensorboard
        if args.tsboard:
            pass
            # writer.add_scalar(f"Test accuracy:Share{args.dataset}, {args.fed}", acc_test, iter)
            # writer.add_scalar(f"Test loss:Share{args.dataset}, {args.fed}", loss_test, iter)
            # if defense_info is not None:
            #     writer.add_scalar(f"Defense/predictor_loss", defense_info['predictor_loss'], iter)
            #     for c, w in enumerate(defense_info['trust_weights']):
            #         writer.add_scalar(f"Defense/trust_weight_cluster_{c}", w, iter)
    
    
    plt.plot(range(args.rounds),result_acc)
    plt.show()
    # writer.close()