import copy
import torch
import random
import numpy as np
import math
import cvxpy as cp
#import scipy.special as sc

def FedAvg(w, args):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        tmp = torch.zeros_like(w[0][k], dtype = torch.float32).to(args.device)
        for i in range(len(w)):
            tmp += w[i][k]
        tmp = torch.true_divide(tmp, len(w))
        w_avg[k].copy_(tmp)
    return w_avg


def FedAvg_Byzantine(w, args, P, wt, G, j, Hh):
    w_avg = copy.deepcopy(w[0])
    B = 1e+6
    N0 = 0 #1e-7
    q = np.zeros((args.num_users-1,1))
    K = args.num_users-1
    gamma, phi = calculate_Phi(w, args, q, wt)
    
    
    H = np.sqrt(np.max(gamma))
    
    min_h = 1
    for i in range(K):
        if  Hh[i][j] > 0.1 and G[i][j] < min_h:
            min_h = G[i][j]
    
    zeta = pow(P[0],0.5)*K/H*min_h 
    
    noise_p = pow(B*N0/2,0.5)*args.lr/zeta
            
    for k in w_avg.keys():
        tmp = torch.zeros_like(w[0][k], dtype = torch.float32).to(args.device)
        tmp += wt[k]
        for i in range(len(w)):
            if i < args.num_byz:
                if args.attack == 'omni':
                    tmp += torch.true_divide(-w[i][k] + wt[k], np.sqrt(gamma[i][0])/pow(2*P[i],0.5)/G[i][j]*zeta)
                    #tmp += torch.true_divide(w[i][k]-wt[k], K)
                elif args.attack == 'gaussian':
                    tmp2 = torch.randn_like(tmp) - torch.ones_like(tmp)
                    tmp2_np = tmp2.numpy()
                    tmp += torch.true_divide(tmp2, np.linalg.norm(tmp2_np)/pow(2*P[i],0.5)/G[i][j]*zeta)
                else:
                    tmp += torch.true_divide(w[i][k]-wt[k], np.sqrt(gamma[i][0])/pow(2*P[i],0.5)/G[i][j]*zeta)
            elif i < args.num_users-1 and Hh[i][j] > 0.1:
                tmp += torch.true_divide(w[i][k]-wt[k], K)
        noise = torch.randn_like(tmp) * noise_p
        tmp = tmp + noise
        w_avg[k].copy_(tmp)
        
    return w_avg


def Secure_aggregation(w, args, idxs_users, distance, P,  wt, reputation, G, j, Hh, q):
    w_avg = copy.deepcopy(w[0])
    
    g_avg = copy.deepcopy(w[0])
    g_tmp = copy.deepcopy(w[0])
    
    
    B = 1e+6
    N0 = 10**(-7)
    L = 10
    
    gamma, phi = calculate_Phi(w, args, q, wt)
    
    H = np.sqrt(np.max(gamma))
    
    varpi = np.zeros((args.num_users-1,1))
    
    
    for i in range(args.num_users-1):
        varpi[i][0] = args.V*L*args.lr**3*B*N0*H/2/(1-L*args.lr)/P[i]/G[i][j]**2
        if Hh[i][j] <= 0.1:
            phi[i][0] = 0
        
    
    if args.optw == 'opt' and j >= 2:
        sorted_idx_ct = sorted(range(len(reputation)), key = lambda x:reputation[x], reverse=False)        
        sorted_idx_ct_np = np.array(sorted_idx_ct) 
        alpha = weight_update(args, phi, varpi, sorted_idx_ct_np)    
    else:
        alpha = np.ones(args.num_users-1)/(args.num_users-1)
    
    
    
    eq_h = np.zeros(args.num_users-1)
    for i in range(args.num_users-1):
        eq_h[i] = pow(P[i],0.5)*G[i][j]/(alpha[i]+1e-9)
    
    sorted_idx = sorted(range(len(eq_h)), key = lambda x:eq_h[x], reverse=False)
    if args.clustering == 'random':
        sorted_idx = random.sample(sorted_idx, len(sorted_idx))
    sorted_idx_np = np.array(sorted_idx) 
    
    num = int((args.num_users-1)/args.num_cluster)
    cluster = np.zeros((num,args.num_cluster))
    
    
    grad = np.empty(shape=[0,0],dtype=float)
    for k in w_avg.keys():
        tmp0 = torch.zeros_like(w[0][k], dtype = torch.float32).to(args.device)    
        cnt = args.num_users-1
        tmp0 += torch.true_divide(w[cnt][k]-wt[k], args.num_users)
        ntmp = tmp0.cpu().numpy()
        grad = np.append(grad,ntmp)
        g_avg[k].copy_(tmp0)
    
    for s in range(args.num_cluster):
        
        cluster[:,s] = sorted_idx_np[s*num:(s+1)*num]
        
        noise_p = 0
        flag_min = 10086
        grad1 = np.empty(shape=[0,0],dtype=float)

        for k in w_avg.keys():
            tmp = torch.zeros_like(w[0][k], dtype = torch.float32).to(args.device)
            tmp1 = torch.zeros_like(w[0][k], dtype = torch.float32).to(args.device)
            for i in range(num):
                cnt = int(cluster[i][s])
                if cnt < args.num_byz:
                    if args.attack == 'omni':
                        tmp1 += torch.true_divide(-w[cnt][k] + wt[k], np.sqrt(gamma[cnt][0])/H/pow(2,0.5)/G[cnt][j])
                    elif args.attack == 'gaussian':
                        tmp2 = torch.randn_like(tmp1) - torch.ones_like(tmp1)
                        tmp2_np = tmp2.numpy()
                        tmp1 += torch.true_divide(tmp2, np.linalg.norm(tmp2_np)/pow(2,0.5)/G[cnt][j]/H)
                    else:
                        tmp1 += torch.true_divide(w[cnt][k]-wt[k], np.sqrt(gamma[cnt][0])/pow(2,0.5)/G[cnt][j]/H)
                elif Hh[cnt][j] > 0.1 and alpha[cnt] > 0:
                    tmp += torch.true_divide(w[cnt][k]-wt[k], 1/alpha[cnt])
                    if eq_h[cnt] < flag_min:
                        flag_min = eq_h[cnt]
            if flag_min == 10086:
                noise_p = 0
            else:
                noise_p = pow(B*N0/2,0.5)*H/flag_min*args.lr/pow(P[cnt],0.5)
                tmp += torch.true_divide(tmp1, flag_min)
            noise = torch.randn_like(tmp) * noise_p
            tmp = tmp + noise
            g_tmp[k].copy_(tmp)
            
            ntmp = tmp.cpu().numpy()
            grad1 = np.append(grad1,ntmp)
        
        if np.dot(grad,grad1)> 0: #/np.linalg.norm(grad)/np.linalg.norm(grad1)
            for k in w_avg.keys():
                tmp = torch.zeros_like(w[0][k], dtype = torch.float32).to(args.device)
                tmp = g_avg[k] + g_tmp[k]
                g_avg[k].copy_(tmp)
                
            for i in range(num):
                cnt = int(cluster[i][s])
                reputation[cnt] += alpha[cnt] * gamma[cnt][0]
                q[cnt] = max(q[cnt] + 0.8*H**2 - gamma[cnt][0], 0)
        else:
            for i in range(num):
                cnt = int(cluster[i][s])
                if args.attack == 'omni':
                    amp = 2
                elif args.attack == 'gaussian':
                    amp = 2
                else:
                    amp = 2
                reputation[cnt] -= amp*alpha[cnt]* gamma[cnt][0]
                
        for k in w_avg.keys():
            tmp = torch.zeros_like(w[0][k], dtype = torch.float32).to(args.device)
            tmp = wt[k] + g_avg[k]
            w_avg[k].copy_(tmp)
        

    return w_avg, reputation, q


def weight_update(args, phi0, varpi0, index):
      
    K = args.num_users-1
    M = args.num_cluster
    K1 = K-args.num_byz
    alpha_n = np.ones((K1,1))/K1
    num = K / M
    
    bar_N = int(math.floor(M-args.num_byz/num))
    e_n = np.ones((bar_N+1,K1))/8.6
    
    tau = 20
    
    phi = np.zeros((K1,1))
    varpi = np.zeros((K1,1))
    
    for k in range(K1):
        phi[k][0] = phi0[index[k+args.num_byz]][0]
        varpi[k][0] = np.sqrt(varpi0[index[k+args.num_byz]][0])
    

    
    for cnt in range(16):
        # Construct the problem.
        alpha = cp.Variable((K1,1))
        u = cp.Variable((bar_N+1,1))
        l = cp.Variable((bar_N+1,1))
        e = cp.Variable((bar_N+1,K1))
        p = cp.Variable((2*bar_N+2,2*K1))
        
        obj = 0
        
        
        for i in range(bar_N):
            for k in range(K1):
                obj += -(alpha_n[k][0]+e_n[i][k])*(alpha[k][0]+e[i][k])/2*phi[k][0]+(alpha[k][0]-e[i][k])**2/4*phi[k][0]
            obj += u[i][0]**2
        obj += tau* cp.sum(p)
        
        objective = cp.Minimize(obj)
        
        constraint_C4 = [-u[i][0]+varpi[k][0]*(1/4*(alpha[k][0]+e[i][k])**2-1/2*(alpha_n[k][0]-e_n[i][k])*(alpha[k][0]-e[i][k])+1/4*(alpha_n[k][0]-e_n[i][k])**2) <= 0 for i in range(bar_N+1) for k in range(K1)]
        constraint_C5 = [-l[i][0]+varpi[k][0]*(alpha[k][0]/e_n[i][k]-alpha_n[k][0]/(e_n[i][k])**2*(e[i][k]-e_n[i][k])) >= 0 for i in range(bar_N+1) for k in range(K1) ]
        constraint_C6 = [u[i][0]-l[i-1][0] <= 0 for i in range(1,bar_N+1)]
        constraint_C10 = [e[i][k]**2-e[i][k]-p[i][k] <= 0 for i in range(bar_N+1) for k in range(K1)]
        constraint_C11 = [-p[i][k]-(2*e_n[i][k]-1)*e[i][k]+e_n[i][k]**2 <= 0 for i in range(bar_N+1) for k in range(K1)]

        constraints = [cp.sum(alpha) == 1, alpha >= 1e-5, alpha <= 0.3, e >= 1e-6, p >= 0, cp.sum(e,axis=0) == 1, cp.sum(e[0:bar_N-1][:],axis=1) == num, cp.sum(e[bar_N][:]) == K1-bar_N*num]#
        prob = cp.Problem(objective, constraints +  constraint_C4 + constraint_C5 + constraint_C6 + constraint_C10 + constraint_C11)
        flag = 0
        try: 
            prob.solve(solver=cp.MOSEK)
            #print(result)
        except:
            print('error')
            flag = 1
        
        if flag == 1:
            break
        
        if (prob.status != 'optimal'):
            print('error')
            break
        if tau<1e6:
            tau=tau*8
        
        alpha_n = alpha.value
        e_n = e.value 
    alphat = np.zeros(K)
    for i in range(K1):
        alphat[index[i+args.num_byz]] = alpha_n[i][0]
        
    return alphat


def calculate_Phi(w, args, q, wt):
    K = args.num_users-1
    
    gamma = np.zeros((K,1))
    
    w_avg = copy.deepcopy(w[0])
    
    for k in w_avg.keys():
        tmp = torch.zeros_like(w[0][k], dtype = torch.float32).to(args.device)
        for i in range(len(w)):
            if i < args.num_byz:
                tmp = -w[i][k] + wt[k]
                tmp_np = tmp.cpu().numpy()
                gamma[i][0] += (np.linalg.norm(tmp_np))**2
            elif i < args.num_users-1:
                tmp = -w[i][k] + wt[k]
                tmp_np = tmp.cpu().numpy()
                gamma[i][0] += (np.linalg.norm(tmp_np))**2
    
    phi = (q + args.V*np.ones((K,1)))*gamma

    return gamma, phi

