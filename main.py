import torch
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

import copy
import numpy as np
import random
from tqdm import trange
import matplotlib.pyplot as plt


from utils.distribute import uniform_distribute, train_dg_split
from utils.sampling import iid, noniid
from utils.options import args_parser
from src.update import ModelUpdate
from src.nets import MLP, CNN_v1, CNN_v2
from src.strategy import FedAvg, FedAvg_Byzantine, Secure_aggregation
from src.test import test_img, test_ind_img
from src.generate import generate_clients
from src.defense import SketchedAirDefense, get_model_flattened_dim, Sketched_Defense_Aggregation

writer = SummaryWriter()

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
        net_glob = CNN_v2(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNN_v1(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    
    print(net_glob)
    net_glob.train()

    # Initialize Sketched-AirDefense if enabled
    defense_module = None
    if args.defense_method == 'sketched':
        model_dim = get_model_flattened_dim(net_glob)
        print(f"[Sketched-AirDefense] Model dimension: {model_dim}")
        print(f"[Sketched-AirDefense] Sketch dimension: {args.sketch_dim}")
        print(f"[Sketched-AirDefense] Window size: {args.window_size}")
        print(f"[Sketched-AirDefense] Number of clusters: {args.num_cluster}")
        defense_module = SketchedAirDefense(args, model_dim, args.device)

    # copy weights
    w_glob = net_glob.state_dict()
        
    # initialization
    initialization_stage = ModelUpdate(args=args, dataset=dataset, idxs=set(dg_idx), rnds=0, data_size = 20, flag_byz = 1)
    w_glob, _ = initialization_stage.train(local_net = copy.deepcopy(net_glob).to(args.device), net = copy.deepcopy(net_glob).to(args.device))
    net_glob.load_state_dict(w_glob)

    
    w_locals = [w_glob for i in range(args.num_users)]
    
    num=0
    
   
    
    distance, P, H, G = generate_clients(args)
    
    
    wt =  net_glob.state_dict()
    
    reputation = np.zeros(args.num_users-1)
    q = np.zeros(args.num_users-1)
    alpha = np.ones(args.num_users-1)/(args.num_users-1)
    
    for iter in trange(args.rounds):
        
        num += 1
        idxs_users = np.array(range(args.num_users))
            

        for idx in idxs_users:
            
            flag = 1
            if idx <= args.num_byz-1 and args.attack == 'label':
                flag = 0
            # Local update
            local = ModelUpdate(args=args, dataset=dataset, idxs=set(list(dict_users[idx])), rnds=num, data_size=int(sumdatasize*alph[idx]), flag_byz=flag)
            
            w, loss = local.train(local_net = copy.deepcopy(net_glob).to(args.device), net = copy.deepcopy(net_glob).to(args.device))
            
            
            w_locals[idx] = copy.deepcopy(w)
              
                
                
        # update global weights
        defense_info = None
        if args.defense_method == 'sketched' and defense_module is not None:
            # Use Sketched-AirDefense with physical layer parameters
            w_glob, reputation, q, defense_info = Sketched_Defense_Aggregation(
                w_locals, args, wt, defense_module, num-1,
                P, G, H, reputation, q
            )
        elif args.trans == 'byzantine':
            w_glob =  FedAvg_Byzantine(w_locals, args, P, wt, G, num-1, H)
        elif args.trans == 'proposed':
            w_glob, reputation, q = Secure_aggregation(w_locals, args, idxs_users, distance, P,  wt, reputation, G, num-1, H, q)
        else:
            w_glob = FedAvg(w_locals, args)
            
        
        wt = copy.deepcopy(w_glob)
        
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        
        acc_test, loss_test = test_img(net_glob, dataset_test, args)

        acc_ind = []
        for idx in idxs_users:
            tmp = test_ind_img(net_glob, dataset, args, set(list(dict_users[idx])))
            acc_ind.append(float(tmp))

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
            writer.add_scalar(f"Test accuracy:Share{args.dataset}, {args.fed}", acc_test, iter)
            writer.add_scalar(f"Test loss:Share{args.dataset}, {args.fed}", loss_test, iter)
            if defense_info is not None:
                writer.add_scalar(f"Defense/predictor_loss", defense_info['predictor_loss'], iter)
                for c, w in enumerate(defense_info['trust_weights']):
                    writer.add_scalar(f"Defense/trust_weight_cluster_{c}", w, iter)
    
    
    plt.plot(range(args.rounds),result_acc)
    plt.show()
    writer.close()