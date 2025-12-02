import torch
import numpy as np
import math


def generate_clients(args):
    distance = np.random.rand(args.num_users)
    distance = distance * 300 + 200
    P = np.ones(args.num_users)
    P = P * 10**(args.power/10)
    
    H = np.zeros((args.num_users-1,args.rounds))
    G = np.zeros((args.num_users-1,args.rounds))
    
    for i in range(args.num_users-1):
        for j in range(args.rounds):
            h_i = np.random.normal(0.0, 1.0, size=None)/pow(2,0.5)
            h_j = np.random.normal(0.0, 1.0, size=None)/pow(2,0.5)
            gamma = h_i**2 + h_j**2
            H[i][j] = gamma
            G[i][j] = gamma*pow(distance[i],-1.1)
    
    return distance, P, H, G
    
