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

from cls_NN import *
from cls_single_task import *
from cls_params_table import *
from utils import *
seed = 34
np.random.seed(seed)
torch.manual_seed(seed)


def seperated_by_class_if_needed(expert_data):
    if (len(expert_data) == 2):
        return expert_data
    else:
        task0 = expert_data[expert_data['y'] == 0]
        task1 = expert_data[expert_data['y'] == 1]
    return [task0, task1]


def get_meta_eta(S,G,t):
    beta = random.random()
    eta1 = S/(G*math.pow(t, beta))
    eta2 = math.pow(t, beta)/(6*S*G*(t+1))
    delta_t = (6*S*G)/math.pow(t, beta)
    return eta1,eta2,delta_t

def omega_R_C(R,C):
    return 0.5*(phi_R_C(R+1,C+1)-phi_R_C(R-1,C-1))

def phi_R_C(R,C):
    print("R",R,"C",C)
    if type(R)== torch.Tensor:
        R= torch.squeeze(R).detach().numpy()
        C = torch.squeeze(C).detach().numpy()
    # R = round(R, 3)
    # C = round(C, 3)
    # print("Now_R", R, "C", C)
    return np.exp((R*R)/(3*C))

def updata_weights_among_experts(activate_experts):
    total_omega =0;
    for expert in activate_experts:
        total_omega += omega_R_C(expert.R,expert.C)
        print("total_omega",total_omega)
    for expert in activate_experts:
        expert.p =  torch.tensor(omega_R_C(expert.R,expert.C)/total_omega)


def meta_update_for_experts(t, d_feature,
                experts, num_neighbors,K,Kq,
                num_iterations, buffer, inner_steps, pd_updates, meta_batch,
                eps, xi,eta_1, eta_2):
    # print("experts",experts)
    meta_eta_1,meta_eta_2,delta_t = get_meta_eta(experts[0].S,experts[0].G,t)


    for iter in range(1, num_iterations + 1):

        for expert in experts:
            net = expert.net
            lamb = expert.lamb
            expert_data = copy.deepcopy(expert.data)
            expert_data = seperated_by_class_if_needed(expert_data)
            expert_eta = expert.eta

            weights = list(net.parameters())

            try:
                lamb = torch.tensor([copy.deepcopy(lamb)], requires_grad=True, dtype=torch.float)
            except:
                lamb = lamb.clone()
        # currently no use
        # if len(buffer) <= meta_batch:
        #     batch = copy.deepcopy(buffer)
        # else:
        #     batch = random.sample(buffer, meta_batch)
        # print("len expert_data", len(expert.data))
        # print("len expert_data", type(expert.data))
        # print("expert_data",expert.data)
        # print("^^^",expert.data[expert.data['y']==0])
        # print("^^^", expert.data[expert.data['y']==1])
        # asd
        # print("expert_data",expert.data)





    # batch = copy.deepcopy(buffer)


            meta_loss = 0

            task = expert_data
            t_loss, t_fair, t_acc, t_dp, t_eop, t_disc, t_cons = cal_loss_and_fairness(t, d_feature, net, lamb, task,
                                                                                       K, Kq, num_neighbors,
                                                                                       inner_steps, pd_updates,
                                                                                       expert_eta, eps, xi)
            if type(t_fair) is not str:  ########## @@@@@@@@@@@@@
                meta_loss = (t_loss + lamb * t_fair - (delta_t * meta_eta_2 / 2) * (lamb ** 2))  ########## @@@@@@@@@@@@@
            else:
                meta_loss = (t_loss - (delta_t / 2) * (lamb ** 2)) ########## @@@@@@@@@@@@@

            meta_loss = meta_loss + xi * (net.e_norm(weights)) ########## @@@@@@@@@@@@@
            expert.meta_loss = meta_loss
            expert.ft = t_loss + lamb

        updata_weights_among_experts(experts)
        meta_loss =0;
        for expert in experts:
            meta_loss += (expert.meta_loss)*(expert.p)
        meta_grads = torch.autograd.grad(meta_loss, weights, retain_graph=True)
        temp_weights = [w.clone() for w in weights]
        weights = [w - meta_eta_1 * g for w, g in zip(temp_weights, meta_grads)]
        weights_norm = net.e_norm(weights) ########## @@@@@@@@@@@@@
        if weights_norm > 10: ########## @@@@@@@@@@@@@
            weights = list(nn.parameter.Parameter(item / weights_norm) for item in weights)
        else:
            weights = list(nn.parameter.Parameter(item) for item in weights)
        weights = list(nn.parameter.Parameter(item) for item in weights)

        net.assign(weights)

        grad_lamb = torch.autograd.grad(meta_loss, lamb)
        lamb = lamb + meta_eta_2 * grad_lamb[0]
        if lamb.item() < 0: ########## @@@@@@@@@@@@@
            lamb = torch.tensor([0], requires_grad=True, dtype=torch.float)

        # update RC for each expert:
        for expert in experts:
            # net = expert.net
            # lamb = expert.lamb
            expert_data = copy.deepcopy(expert.data)
            expert_data = seperated_by_class_if_needed(expert_data)
            expert_eta = expert.eta

            weights = list(net.parameters())

            try:
                lamb = torch.tensor([copy.deepcopy(lamb)], requires_grad=True, dtype=torch.float)
            except:
                lamb = lamb.clone()
            # t_loss, t_fair, t_acc, t_dp, t_eop, t_disc, t_cons = cal_loss_and_fairness(t, d_feature, net, lamb, task,
            #                                                                        K, Kq, num_neighbors,
            #                                                                        inner_steps, pd_updates,
            #                                                                        expert_eta, eps, xi)
            loss_val, fair_val, accuracy_val, dp_val, eop_val, discrimination_val, consistency_val = validate_performance(
                t, d_feature, net, lamb, task,
                K, Kq, num_neighbors,
                inner_steps, pd_updates,
                eta_1, eta_2, eps, xi)
            meta_ft = loss_val+fair_val

            expert.R = expert.R +meta_ft-expert.ft
            expert.C = expert.C +abs(meta_ft-expert.ft)






    return net, lamb


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
         eta_1, eta_2, eps, xi):
    now = datetime.datetime.now()
    exp_name = now.strftime("\%Y-%m-%d-%H-%M-%S-Ours-" + dataset)
    save_folder = save + exp_name
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    val_save_path = save_folder + r"\val.txt"
    with open(save_folder + r'\hyper-parameters.txt', 'wb') as f:
        t = hyper_params(d_feature, lamb, dataset,
                         K, Kq, val_batch_size, num_neighbors,
                         num_iterations, inner_steps, pd_updates, meta_batch,
                         eta_1, eta_2, eps, xi)
        f.write(t)

    return val_save_path


def pdrftml(d_feature, lamb, tasks, data_path, dataset, save,
            K, Kq, val_batch_size, num_neighbors,
            num_iterations, inner_steps, pd_updates, meta_batch,
            eta_1, eta_2, eps, xi):
    val_save_path = prep(save, d_feature, lamb, dataset,
                         K, Kq, val_batch_size, num_neighbors,
                         num_iterations, inner_steps, pd_updates, meta_batch,
                         eta_1, eta_2, eps, xi)
    net = NN(d_feature)
    # weights = list(net.parameters())
    lamb = copy.deepcopy(lamb)
    buffer = []
    T = len(tasks)
    res = []
    dataset_full_path = data_path + '\\' + dataset
    set_A = []
    # return
    active_experts=[]

    for t in range(1, T + 1):
        if t==1:
            set_A = initialize_set_A(data_path=dataset_full_path,task_length=T,lamb = lamb,net=net,eps=eps,d_feature=d_feature)
            active_experts= set_A
        else:
            set_A,active_amount = update_data_in_A_at_t(A=set_A,t=t,lamb = lamb,net=net)
            active_experts = set_A[:active_amount]
        start_time = time.time()
        # print("task" + str(t))
        # task0 = pd.read_csv(data_path + '\\' + dataset + r'\task' + str(t) + r'\task' + str(t) + '_neg.csv')
        # task1 = pd.read_csv(data_path + '\\' + dataset + r'\task' + str(t) + r'\task' + str(t) + '_pos.csv')
        task0 = pd.read_csv(data_path + '\\' + dataset + r'\task' + str(t)  + r'\neg.csv')
        task1 = pd.read_csv(data_path + '\\' + dataset + r'\task' + str(t) + r'\pos.csv')

        task = [task0, task1]
        buffer.append(task)

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
        res.append([loss_val.item(), fair_val, accuracy_val, dp_val, eop_val, discrimination_val, consistency_val, cost_time])
        ###########################################################################################################################################################

        # meta-train
        # for experts in active_experts:

        new_net, new_lamb = meta_update_for_experts(t, d_feature, active_experts,
                                         num_neighbors,K, Kq,
                                        num_iterations, buffer, inner_steps, pd_updates, meta_batch,
                                        eps, xi,eta_1, eta_2)

        net = new_net
        lamb = new_lamb

    with open(val_save_path, 'wb') as f:
        pickle.dump(res, f)
