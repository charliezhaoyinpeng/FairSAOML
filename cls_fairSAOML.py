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

from cls_single_task import *
from cls_params_table import *
from utils import *
seed = 34
np.random.seed(seed) #全换成0或者保持这个都试一下
torch.manual_seed(seed)





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
    if R <0: R=0
    return np.exp((R*R)/(3*C))

def updata_P_in_experts(experts):
    total_W =0;
    for expert in experts:
        total_W += W_R_C(expert.R,expert.C)
        # print("total_omega",total_W)
    for expert in experts:
        expert.p =  torch.tensor(W_R_C(expert.R,expert.C)/total_W)
        # print("expert.p",expert.p)


def meta_update_for_experts(t, d_feature,
                meta_net,meta_lamb,experts,active_amount, num_neighbors,K,Kq,
                num_iterations, inner_steps, pd_updates,
                eps, xi,radius,meta_eta_1,meta_eta_2,delta):
    # delta = 0.01 # need change later 10 100
    # print("%%%%",meta_eta_1,meta_eta_2)
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
            expert_level_supporting(t, d_feature, expert, task,K, Kq, num_neighbors,inner_steps, pd_updates,expert_eta, eps, xi,radius)
        meta_loss = 0;
        meta_grad_weights=None
        meta_grad_lambda = None
        for expert in experts:
            # lamb = expert.lamb
            query_grad_weights,query_grad_lambda = expert_level_quering(t, d_feature, expert, task,K, Kq, num_neighbors,inner_steps, pd_updates,expert_eta, eps, xi,delta,radius)
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

        # print("query_grad_weights",query_grad_weights)
        # print("query_grad_lambda", query_grad_lambda)


            #expert_query_loss = expert_query_loss #+ xi * (net.e_norm(weights)) ########## @@@@@@@@@@@@
        # weights = list(meta_net.parameters())

        # print(meta_net)

        # temp = [w.clone() for w in temp_weights]

        temp_weights = [(w - meta_eta_1 * g) for w, g in zip(temp_weights, meta_grad_weights)]
        weights_norm = meta_net.e_norm(temp_weights)

        if weights_norm > radius: ########## @@@@@@@@@@@@@ 改成一个参数 radius
            temp_weights = list(nn.parameter.Parameter(item / weights_norm) for item in temp_weights)
        else:
            temp_weights = list(nn.parameter.Parameter(item) for item in temp_weights)
        temp_weights = list(nn.parameter.Parameter(item) for item in temp_weights)
        temp_weights = list(nn.parameter.Parameter(item) for item in temp_weights)
        # print("!!!!****", weights)


        # meta_net.assign(weights)
        # print(lamb)
        # print(meta_lamb,meta_loss)
        # print(temp_lambda)
        # print("ok",grad_lamb)
        temp_lambda = temp_lambda + meta_eta_2 * meta_grad_lambda[0]

        # grad_lamb = torch.autograd.grad(meta_loss, meta_lamb, retain_graph=False)
        # meta_lamb = meta_lamb + meta_eta_2 * grad_lamb[0]
        # print("ok")

        # meta_lamb = torch.tensor(temp_lambda)
        # print("meta lamb update",meta_lamb)
        # if meta_lamb.item() < 0: ########## @@@@@@@@@@@@@
        #     meta_lamb = torch.tensor([0], requires_grad=True, dtype=torch.float)

            # update RC for each expert:

    count =1
    for expert in experts:
        # print("******"+str(count)+" expert")
        count = count+1
        # print("123",list(expert.net.parameters()))
        # print("!123",list(meta_net.parameters()))
        update_expert_RC(expert, temp_weights,temp_lambda, eps, Kq,d_feature)
    # print("pass all")
    meta_net.assign(temp_weights)
    meta_lamb = temp_lambda

    return meta_net, meta_lamb



# def meta_update_back_up(t, d_feature, net, lamb,
#                 K, Kq, num_neighbors,
#                 num_iterations, buffer, inner_steps, pd_updates, meta_batch,
#                 eta_1, eta_2, eta_3, eta_4, delta, eps, xi):
#     weights = list(net.parameters())
#
#     try:
#         lamb = torch.tensor([copy.deepcopy(lamb)], requires_grad=True, dtype=torch.float)
#     except:
#         lamb = lamb.clone()
#
#     if len(buffer) <= meta_batch:
#         batch = copy.deepcopy(buffer)
#     else:
#         batch = random.sample(buffer, meta_batch)
#
#     # batch = copy.deepcopy(buffer)
#
#     for iter in range(1, num_iterations + 1):
#         meta_loss = 0
#         for i in range(len(batch)):
#             task = batch[i]
#             t_loss, t_fair, t_acc, t_dp, t_eop, t_disc, t_cons = cal_loss_and_fairness(t, d_feature, net, lamb, task,
#                                                                                        K, Kq, num_neighbors,
#                                                                                        inner_steps, pd_updates,
#                                                                                        eta_1, eta_2, eps, xi)
#             if type(t_fair) is not str:  ########## @@@@@@@@@@@@@
#                 meta_loss += (t_loss + lamb * t_fair - (delta * eta_4 / 2) * (lamb ** 2))  ########## @@@@@@@@@@@@@
#             else:
#                 meta_loss += (t_loss - (delta / 2) * (lamb ** 2))                   ########## @@@@@@@@@@@@@
#
#         meta_loss = meta_loss + xi * (net.e_norm(weights)) ########## @@@@@@@@@@@@@
#
#         meta_grads = torch.autograd.grad(meta_loss, weights, retain_graph=True)
#         temp_weights = [w.clone() for w in weights]
#         weights = [w - eta_3 * g for w, g in zip(temp_weights, meta_grads)]
#         weights_norm = net.e_norm(weights) ########## @@@@@@@@@@@@@
#         if weights_norm > 10: ########## @@@@@@@@@@@@@
#             weights = list(nn.parameter.Parameter(item / weights_norm) for item in weights)
#         else:
#             weights = list(nn.parameter.Parameter(item) for item in weights)
#         weights = list(nn.parameter.Parameter(item) for item in weights)
#
#         net.assign(weights)
#
#         grad_lamb = torch.autograd.grad(meta_loss, lamb)
#         lamb = lamb + eta_4 * grad_lamb[0]
#         if lamb.item() < 0: ########## @@@@@@@@@@@@@
#             lamb = torch.tensor([0], requires_grad=True, dtype=torch.float)
#
#     return net, lamb


def prep(save, d_feature, lamb, dataset,
         K, Kq, val_batch_size, num_neighbors,
         num_iterations, inner_steps, pd_updates, meta_batch,
         eta_1, eta_2, eps, xi,radius):
    now = datetime.datetime.now()
    exp_name = now.strftime("\%Y-%m-%d-%H-%M-%S-Ours-" + dataset)
    save_folder = save + exp_name
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    val_save_path = os.path.join(save_folder + "val.txt")
    with open(os.path.join(save_folder + 'hyper-parameters.txt'), 'wb') as f:
        t = hyper_params(d_feature, lamb, dataset,
                         K, Kq, val_batch_size, num_neighbors,
                         num_iterations, inner_steps, pd_updates, meta_batch,
                         eta_1, eta_2, eps, xi,radius)
        f.write(t)

    return val_save_path

def print_experts(t,experts):
    print("current_time ",t)
    for expert in experts:
        expert.print_expert();
def fairSAOML(d_feature, lamb, tasks, data_path, dataset, save,
            K, Kq, val_batch_size, num_neighbors,
            num_iterations, inner_steps, pd_updates, meta_batch,
            eta_1, eta_2, eps, xi,radius,meta_eta_1,meta_eta_2,delta):
    val_save_path = prep(save, d_feature, lamb, dataset,
                         K, Kq, val_batch_size, num_neighbors,
                         num_iterations, inner_steps, pd_updates, meta_batch,
                         eta_1, eta_2, eps, xi,radius)
    # net = NN_init_0(d_feature)  # init by setting weights to 0
    net = NN(d_feature) #random init
    # weights = list(net.parameters())
    # lamb = copy.deepcopy(lamb)
    T = len(tasks)
    dp = []
    eop=[]
    acc=[]
    loss=[]
    dataset_full_path = os.path.join(data_path , dataset)
    # set_A = []
    set_U = []

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
                                                                                                                       eta_1, eta_2, eps, xi)

        cost_time = time.time() - start_time
        print("Val-Task %s/%s: acc:%s ;dp:%s; eop:%s; disc:%s" % (
            t, T, np.round(accuracy_val, 10), np.round(dp_val, 10), np.round(eop_val, 10),
            np.round(discrimination_val, 10)))

        # torch.save(net.state_dict(), model_save_path)
        # res.append([loss_val.item(), fair_val, accuracy_val, dp_val, eop_val, discrimination_val, consistency_val, cost_time])
        dp.append(np.round(dp_val, 10))
        eop.append(np.round(eop_val, 10))
        acc.append(np.round(accuracy_val, 10))
        loss.append(np.round(loss_val.detach().numpy(), 10))


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


        print_experts(t,set_U)

        ###########################################################################################################################################################

        # meta-train
        # for experts in active_experts:

        new_net, new_lamb = meta_update_for_experts(t, d_feature, net,lamb,set_U,active_amount,
                                         num_neighbors,K, Kq,
                                        num_iterations,inner_steps, pd_updates,
                                        eps, xi,radius,meta_eta_1,meta_eta_2,delta)

        net = new_net
        lamb = new_lamb

    with open(val_save_path, 'wb') as f:
        pickle.dump([dp,eop,acc,loss], f)
    return dp,eop,acc,loss
