from __future__ import division
import torch, os
import torch.nn as nn
import random
import math
import copy, time, pickle, datetime
import pandas as pd
import numpy as np
from numpy import linalg as LA
from eval_metrics_cls import *
# fair strong adaptive online meta learning fairsaoml
from cls_NN import *
from draw import draw
from cls_single_task import *
from cls_params_table import *
from utils import *
seed = 117 #34,27 53 57 108 109 119 175 183 191 230 acc 43 ((47 117
np.random.seed(seed) #全换成0或者保持这个都试一下
torch.manual_seed(seed)

# def set_seed(seed1):
#     global seed
#     seed= seed  # 34,27 53 57 108 109 119 175 183 191 230 acc 43
#     np.random.seed(seed1)  # 全换成0或者保持这个都试一下
#     torch.manual_seed(seed1)



def get_meta_eta(S,G,t):
    beta = random.random()
    eta1 = S/(G*math.pow(t, beta))
    eta2 = math.pow(t, beta)/(6*S*G*(t+1))
    delta_t = (6*S*G)/math.pow(t, beta)
    return eta1,eta2,delta_t

def W_R_C(R,C):
    return 0.5*(phi_R_C(R+1,C+1)-phi_R_C(R-1,C-1))

def phi_R_C(R,C):
    # print("R",R,"C",C)
    if type(R)== torch.Tensor:
        R= torch.squeeze(R).detach().numpy()
        C = torch.squeeze(C).detach().numpy()
    if R ==0 and C== 0:
        return 0
    # R = round(R, 3)
    # C = round(C, 3)
    # print("Now_R", R, "C", C)
    if R <0: return 1
    return np.exp((R*R)/(3*C))

def updata_P_in_experts(experts):
    total_W =0;
    for expert in experts:
        expert.w = W_R_C(expert.R,expert.C)
        total_W += expert.w
        # print("total_omega",total_W)
    for expert in experts:
        expert.p =  torch.tensor(expert.w/total_W)
        # print("expert.p",expert.p)


def meta_update_for_experts(t, d_feature,
                meta_net,meta_lamb,experts,active_amount, num_neighbors,K,Kq,
                num_iterations, inner_steps, pd_updates,
                eps, radius,meta_eta_1,meta_eta_2,delta,shift_time,meta_eta_1_shift,meta_eta_2_shift):
    # delta = 0.01 # need change later 10 100
    # print("%%%%",meta_eta_1,meta_eta_2)
    if t>shift_time:
        meta_eta_1 = meta_eta_1_shift
        meta_eta_2 = meta_eta_2_shift
    temp1= list(meta_net.parameters())

    try:
        temp_weights = [torch.tensor([copy.deepcopy(w)], requires_grad=True, dtype=torch.float) for w in
                        temp1]
    except:
        temp_weights = [w.clone() for w in temp1]

    try:
        temp_lambda = torch.tensor([copy.deepcopy(meta_lamb)], requires_grad=True, dtype=torch.float)
    except:
        temp_lambda = meta_lamb
    print("temp_lambda",float(temp_lambda))

    for iter in range(1, num_iterations + 5):
        # print(type(meta_lamb))

        for expert in experts[:active_amount]:
            expert_data = copy.deepcopy(expert.data)
            task = seperated_by_class_if_needed(expert_data)
            expert_eta = expert.eta

            expert_level_supporting(t, d_feature, expert, task,K, Kq, num_neighbors,inner_steps, pd_updates,expert_eta, eps)
        meta_loss = 0;
        meta_grad_weights=None
        meta_grad_lambda = None
        for expert in experts:
            # lamb = expert.lamb
            query_grad_weights,query_grad_lambda = expert_level_quering(t, d_feature, expert, task,K, Kq, num_neighbors,inner_steps, pd_updates,expert_eta, eps,delta)
            if meta_grad_weights ==None:
                # print(query_grad_weights)
                temp = [w.clone()*expert.p for w in query_grad_weights]
                meta_grad_weights=temp
            else:
                temp = [w.clone() * expert.p for w in query_grad_weights]
                meta_grad_weights += temp
            if meta_grad_lambda == None:
                meta_grad_lambda = torch.mul(query_grad_lambda[0],expert.p)
            else:
                meta_grad_lambda+=torch.mul(query_grad_lambda[0],expert.p)

        temp_weights = [(w - meta_eta_1 * g) for w, g in zip(temp_weights, meta_grad_weights)]
        weights_norm = meta_net.e_norm(temp_weights)

        if weights_norm > radius: ########## @@@@@@@@@@@@@ 改成一个参数 radius
            temp_weights = list(nn.parameter.Parameter(item / weights_norm) for item in temp_weights)
        else:
            temp_weights = list(nn.parameter.Parameter(item) for item in temp_weights)
        temp_weights = list(nn.parameter.Parameter(item) for item in temp_weights)
        temp_weights = list(nn.parameter.Parameter(item) for item in temp_weights)
        if temp_lambda + meta_eta_2 * meta_grad_lambda[0]<=0:
            temp_lambda =temp_lambda*0
        else:
            temp_lambda = temp_lambda + meta_eta_2 * meta_grad_lambda[0]


    count =1
    for expert in experts:
        count = count+1
        update_expert_RC(expert, temp_weights,temp_lambda, eps,d_feature)
    # print("pass all")
    meta_net.assign(temp_weights)
    meta_lamb = temp_lambda

    return meta_net, meta_lamb



def prep(save, d_feature, lamb, tasks, data_path, dataset,
            K, Kq, val_batch_size, num_neighbors,
            num_iterations, inner_steps, pd_updates,
            eta_1, eta_2, eps,radius,meta_eta_1,meta_eta_2,delta,shift_time,eta_1_shift,eta_2_shift,meta_eta_1_shift,meta_eta_2_shift,net_dim):
    now = datetime.datetime.now()
    exp_name = now.strftime("%Y-%m-%d-%H-%M-%S-Ours-" + dataset)
    save_folder = os.path.join(save , exp_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    val_save_path = os.path.join(save_folder, "val.np")
    print(val_save_path)
    with open(os.path.join(save_folder, 'hyper-parameters.txt'), 'wb') as f:
        t = hyper_params(d_feature, lamb, tasks, data_path, dataset,
            K, Kq, val_batch_size, num_neighbors,
            num_iterations, inner_steps, pd_updates,
            eta_1, eta_2, eps,radius,meta_eta_1,meta_eta_2,delta,shift_time,eta_1_shift,eta_2_shift,meta_eta_1_shift,meta_eta_2_shift,net_dim)
        f.write(t)

    return val_save_path,save_folder

def record_print_experts(t,experts):
    print("current_time ",t)
    expert_data=[]
    for expert in experts:
        expert_data.append(expert.print_expert());
    return {t:expert_data}
# def find_seeds(t, d_feature, net, lamb, task,K, val_batch_size, num_neighbors,inner_steps, pd_updates,eta_1, eta_2, eps,shift_time,eta_1_shift,eta_2_shift,seed):
#     seeds=[]
#     while 1:
#         set_seed(seed)
#         print("seed",seed)
#         loss_val, fair_val, accuracy_val, dp_val, eop_val, discrimination_val, consistency_val = validate_performance(t, d_feature, net, lamb, task,K, val_batch_size, num_neighbors,inner_steps, pd_updates,eta_1, eta_2, eps,shift_time,eta_1_shift,eta_2_shift)
#         if accuracy_val<0.4 and  dp_val<0.4 and eop_val<0.4:
#             seeds.append(seed)
#             print("seed ",seed," acc ",accuracy_val," dp_val ",dp_val," eop_val ",eop_val)
#         seed = seed +1
#         print("seed ", seed, " acc ", accuracy_val, " dp_val ", dp_val, " eop_val ", eop_val)
#         if len(seeds)==5:
#             print(seeds)
#             break
def fairSAOML(d_feature, lamb, tasks, data_path, dataset, save,
            K, Kq, val_batch_size, num_neighbors,
            num_iterations, inner_steps, pd_updates,
            eta_1, eta_2, eps,radius,meta_eta_1,meta_eta_2,delta,shift_time,eta_1_shift,eta_2_shift,meta_eta_1_shift,meta_eta_2_shift,net_dim):
    val_save_path,save_folder = prep(save, d_feature, lamb, tasks, data_path, dataset,
            K, Kq, val_batch_size, num_neighbors,
            num_iterations, inner_steps, pd_updates,
            eta_1, eta_2, eps,radius,meta_eta_1,meta_eta_2,delta,shift_time,eta_1_shift,eta_2_shift,meta_eta_1_shift,meta_eta_2_shift,net_dim)
    # net = NN_init_0(d_feature)  # init by setting weights to 0
    net = NN(d_feature,net_dim) #random init

    # weights = list(net.parameters())
    # lamb = copy.deepcopy(lamb)
    T = len(tasks)
    dp = []
    eop=[]
    acc=[]
    loss=[]
    dbc=[]
    loss_plus_dbc=[]
    dataset_full_path = os.path.join(data_path , dataset)
    # set_A = []
    set_U = []
    experts_data=[]
    task0 = pd.read_csv(os.path.join(data_path, dataset, 'task' + str(1), r'neg.csv'))
    task1 = pd.read_csv(os.path.join(data_path, dataset, 'task' + str(1), r'pos.csv'))

    task = [task0, task1]
    # find_seeds(1, d_feature, net, lamb, task,K, val_batch_size, num_neighbors,inner_steps, pd_updates,eta_1, eta_2, eps,shift_time,eta_1_shift,eta_2_shift,seed)
    for t in range(1, T + 1):

        start_time = time.time()
        task0 = pd.read_csv(os.path.join(data_path, dataset ,'task' + str(t)  , r'neg.csv'))
        task1 = pd.read_csv(os.path.join(data_path , dataset , 'task' + str(t) , r'pos.csv'))

        task = [task0, task1]
        # print()

        # evaluation
        ###########################################################################################################################################################
        print("val_batch_size",val_batch_size,"K",K,"defeature",d_feature)
        loss_val, fair_val, accuracy_val, dp_val, eop_val, discrimination_val, consistency_val = validate_performance(t, d_feature, net, lamb, task,
                                                                                                                       K, val_batch_size, num_neighbors,
                                                                                                                       inner_steps, pd_updates,
                                                                                                                       eta_1, eta_2, eps,shift_time,eta_1_shift,eta_2_shift)

        cost_time = time.time() - start_time
        print("Val-Task %s/%s: acc:%s ;dp:%s; eop:%s; disc:%s; loss:%s; dbc:%s;dbc+loss:%s"% (
            t, T, np.round(accuracy_val, 10), np.round(dp_val, 10), np.round(eop_val, 10),
            np.round(discrimination_val, 10),np.round(loss_val.detach().numpy(), 10),np.round(fair_val.detach().numpy(), 10),np.round(loss_val.detach().numpy()+fair_val.detach().numpy(), 10)))

        # torch.save(net.state_dict(), model_save_path)
        # res.append([loss_val.item(), fair_val, accuracy_val, dp_val, eop_val, discrimination_val, consistency_val, cost_time])
        dp.append(np.round(dp_val, 10))
        eop.append(np.round(eop_val, 10))
        acc.append(np.round(accuracy_val, 10))
        loss.append(np.round(loss_val.detach().numpy(), 10))
        dbc.append(np.round(fair_val.detach().numpy(), 10))
        loss_plus_dbc.append(np.round(loss_val.detach().numpy()+fair_val.detach().numpy(), 10))



        try:
            lamb = torch.tensor([copy.deepcopy(lamb)], requires_grad=True, dtype=torch.float)
        except:
            lamb = lamb.clone()


        if t==1:
            set_U = initialize_set_U(data_path=dataset_full_path,task_length=T,lamb = lamb,net=net,eps=eps,d_feature=d_feature)
            active_amount = len(set_U)
        else:

            set_U,active_amount = update_experts_at_t(A=set_U,t=t,lamb = lamb,net=net)
            updata_P_in_experts(set_U)


        experts_data.append(record_print_experts(t,set_U))

        ###########################################################################################################################################################

        # meta-train
        # for experts in active_experts:

        new_net, new_lamb = meta_update_for_experts(t, d_feature, net,lamb,set_U,active_amount,
                                         num_neighbors,K, Kq,
                                        num_iterations,inner_steps, pd_updates,
                                        eps, radius,meta_eta_1,meta_eta_2,delta,shift_time,meta_eta_1_shift,meta_eta_2_shift)

        net = new_net
        lamb = new_lamb

    with open(val_save_path, 'wb') as f:
        pickle.dump({'dp':dp,'eop':eop,'acc':acc,'loss':loss,'dbc':dbc,'dbc_loss':loss_plus_dbc,'experts_data':experts_data}, f)
    draw(save_folder)
    return dp,eop,acc,loss,dbc,loss_plus_dbc


