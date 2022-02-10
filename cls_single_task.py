from __future__ import division
import torch, os
import torch.nn as nn
import random
import math
import copy, time, pickle, datetime
import pandas as pd
import numpy as np
from numpy import linalg as LA
from sklearn.metrics import accuracy_score

from eval_metrics_cls import *
from cls_NN import *


def seperated_by_class_if_needed(expert_data):
    task0 = expert_data[expert_data['y'] == 0]
    task1 = expert_data[expert_data['y'] == 1]

    return [task0, task1]

def sample_data_from_task(task, batch_size, d_feature):
    data = task.sample(batch_size)
    X = data[data.columns[-d_feature:]].copy()
    y = data[["y"]]
    z = data[["z"]]
    return X, y, z


def mean(a):
    return sum(a).to(dtype=torch.float) / len(a)



def validate_performance(t, d_feature, net, lamb, task,
                          K, Kq, num_neighbors,
                          inner_steps, pd_updates,
                          eta_1, eta_2, eps,shift_time,eta_1_shift_1,eta_2_shift_1,eta_1_shift_2,eta_2_shift_2):
    temp_weights = [w.clone() for w in list(net.parameters())]
    if t> 30 and t<=60:
        eta_1 = eta_1_shift_1
        eta_2 = eta_2_shift_1
    elif t>60:
        eta_1 = eta_1_shift_2
        eta_2 = eta_2_shift_2
    try:
        temp_lambda = torch.tensor([copy.deepcopy(lamb)], requires_grad=True, dtype=torch.float)
    except:
        temp_lambda = lamb.clone()

    criterion = nn.BCELoss()
    task0 = task[0]
    task1 = task[1]
    task_df = pd.concat([task0, task1])


    X0_s, y0_s, z0_s = sample_data_from_task(task0, K, d_feature)
    X1_s, y1_s, z1_s = sample_data_from_task(task1, K, d_feature)
    X_s = pd.concat([X0_s, X1_s]).values
    y_s = pd.concat([y0_s, y1_s]).values
    z_s = pd.concat([z0_s, z1_s]).values
    z_bar = np.mean(z_s) * np.ones((len(z_s), 1))

    X_s = torch.tensor(X_s, dtype=torch.float).unsqueeze(1)
    y_s = torch.tensor(y_s, dtype=torch.float).unsqueeze(1)
    ones = torch.tensor(np.ones((len(y_s), 1)), dtype=torch.float).unsqueeze(1)
    z_s = torch.tensor(z_s, dtype=torch.float).unsqueeze(1)
    z_bar = torch.tensor(z_bar, dtype=torch.float).unsqueeze(1)


    for co_update in range(1):
        for step in range(1):
            y_hat = net.parameterised(X_s, temp_weights)
            fair = cal_dbc(torch.squeeze(z_s),torch.squeeze(y_hat))-eps
            loss = (-1.0) * torch.mean(y_s * torch.log(y_hat) + (ones - y_s) * torch.log(ones - y_hat))
            loss = loss / K +temp_lambda * fair
            grad = torch.autograd.grad(loss.sum(), temp_weights, retain_graph=True)
            temp_weights = [w - eta_1 * g for w, g in zip(temp_weights, grad)]


            new_y_hat = net.parameterised(X_s, temp_weights)

            fair = cal_dbc(torch.squeeze(z_s), torch.squeeze(y_hat)) - eps
            loss = (-1.0) * torch.mean(y_s * torch.log(new_y_hat) + (ones - y_s) * torch.log(ones - new_y_hat))
            loss = loss / K + temp_lambda * fair

            grad_lamb = torch.autograd.grad(loss.sum(), temp_lambda)

            temp_lambda = temp_lambda + eta_2 * grad_lamb[0]
            if temp_lambda.item() < 0:
                temp_lambda = torch.tensor([0], requires_grad=True, dtype=torch.float)

    # if Kq < 1: # means for validation purpose
    Kq = round(len(task_df.index) * Kq * 1.0 / 100)

    X_q, y_q, z_q = sample_data_from_task(task_df, Kq, d_feature)
    X_q = X_q.values
    y_q = y_q.values
    z_q = z_q.values

    X_temp = copy.deepcopy(X_q)
    z_temp = copy.deepcopy(z_q)
    y_temp = copy.deepcopy(y_q)

    X_q = torch.tensor(X_q, dtype=torch.float).unsqueeze(1)
    y_q = torch.tensor(y_q, dtype=torch.float).unsqueeze(1)
    z_q = torch.tensor(z_q, dtype=torch.float).unsqueeze(1)

    y_hat = net.parameterised(X_q, temp_weights)

    loss = criterion(y_hat, y_q)

    fair = cal_dbc(torch.squeeze(z_q), torch.squeeze(y_hat))

    y_hat = y_hat.detach().numpy().reshape(len(y_hat), 1)
    y_q = y_q.detach().numpy().reshape(len(y_q), 1)

    input_zy = np.column_stack((z_temp, y_hat))
    z_y_hat_y = np.column_stack((input_zy, y_temp))

    accuracy = accuracy_score(y_hat.round(), y_q)
    dp = cal_dp(input_zy)
    eop = cal_eop(z_y_hat_y)
    discrimination = cal_discrimination(input_zy) * 100






    return loss, fair, accuracy, dp, eop, discrimination


def update_expert_RC(expert,meta_weights,meta_lambda,eps,d_feature):


    expert_data = copy.deepcopy(expert.data)
    task = seperated_by_class_if_needed(expert_data)
    task0 = task[0]
    task1 = task[1]
    task_df = pd.concat([task0, task1])
    X = task_df[task_df.columns[-d_feature:]].copy().values
    y = task_df[["y"]].values
    z = task_df[["z"]].values
    total = len(y)

    criterion = nn.BCELoss()
    # z_bar = np.mean(z_q) * np.ones((len(z_q), 1))

    X = torch.tensor(X, dtype=torch.float).unsqueeze(1)
    y = torch.tensor(y, dtype=torch.float).unsqueeze(1)
    z = torch.tensor(z, dtype=torch.float).unsqueeze(1)
    # z_bar = torch.tensor(z_bar, dtype=torch.float).unsqueeze(1)



    y_hat_meta = expert.net.parameterised(X,meta_weights)

    loss_meta = criterion(y_hat_meta, y)
    loss_meta = float(loss_meta / total)
    fair_meta = float(cal_dbc(torch.squeeze(z), torch.squeeze(y_hat_meta)) - eps)
    meta_ft = loss_meta+fair_meta*meta_lambda

    temp_weights = [w.clone() for w in list(expert.net.parameters())]
    y_hat = expert.net.parameterised(X,temp_weights)
    loss = criterion(y_hat, y)
    loss = float(loss / total)
    fair = float(cal_dbc(torch.squeeze(z), torch.squeeze(y_hat)) - eps)
    expert_ft = loss + fair*expert.lamb

    expert.R = expert.R +meta_ft-expert_ft
    expert.C = expert.C+abs(meta_ft-expert_ft)



def expert_level_supporting(t, d_feature, expert, task,
                          K, Kq, num_neighbors,
                          inner_steps, pd_updates,
                          expert_eta, eps):
    prev_weights = [w.clone() for w in expert.weights]


    try:
        temp_weights = [torch.tensor([copy.deepcopy(w)], requires_grad=True, dtype=torch.float) for w in expert.weights]
    except:
        temp_weights = [w.clone() for w in expert.weights]

    try:
        temp_lambda = torch.tensor([copy.deepcopy(expert.lamb)], requires_grad=True, dtype=torch.float)
    except:
        temp_lambda = expert.lamb.clone()

    task0 = task[0]
    task1 = task[1]

    X0_s, y0_s, z0_s = sample_data_from_task(task0, K, d_feature)
    X1_s, y1_s, z1_s = sample_data_from_task(task1, K, d_feature)
    X_s = pd.concat([X0_s, X1_s]).values
    y_s = pd.concat([y0_s, y1_s]).values
    z_s = pd.concat([z0_s, z1_s]).values


    X_s = torch.tensor(X_s, dtype=torch.float).unsqueeze(1)
    y_s = torch.tensor(y_s, dtype=torch.float).unsqueeze(1)
    ones = torch.tensor(np.ones((len(y_s), 1)), dtype=torch.float).unsqueeze(1)
    z_s = torch.tensor(z_s, dtype=torch.float).unsqueeze(1)


    for co_update in range(pd_updates):
        for step in range(inner_steps):
            y_hat = expert.net.parameterised(X_s, temp_weights)
            fair = cal_dbc(torch.squeeze(z_s),torch.squeeze(y_hat))-eps

            loss = (-1.0) * torch.mean(y_s * torch.log(y_hat) + (ones - y_s) * torch.log(ones - y_hat))
            loss = loss / K+ temp_lambda * fair
            grad = torch.autograd.grad(loss.sum(), temp_weights, retain_graph=True)
            temp_weights = [w - expert_eta * g for w, g in zip(temp_weights, grad)]



            y_hat = expert.net.parameterised(X_s, temp_weights)

            fair = cal_dbc(torch.squeeze(z_s), torch.squeeze(y_hat)) - eps
            loss = (-1.0) * torch.mean(y_s * torch.log(y_hat) + (ones - y_s) * torch.log(ones - y_hat))
            loss = loss / K + temp_lambda * fair

            grad_lamb = torch.autograd.grad(loss.sum(), temp_lambda)

            temp_lambda = temp_lambda + expert_eta * grad_lamb[0]

            if temp_lambda.item() < 0:
                temp_lambda = torch.tensor([0], requires_grad=True, dtype=torch.float)
    weights = list(nn.parameter.Parameter(item) for item in temp_weights)
    weights = list(nn.parameter.Parameter(item) for item in weights)
    expert.weights = weights
    expert.lamb = temp_lambda


def expert_level_quering(t, d_feature, expert, task,
                                K, Kq, num_neighbors,
                                inner_steps, pd_updates,
                                expert_eta, eps,delta):
    try:
        temp_weights = [torch.tensor([copy.deepcopy(w)], requires_grad=True, dtype=torch.float) for w in expert.weights]
    except:
        temp_weights = [w.clone() for w in expert.weights]

    try:
        temp_lambda = torch.tensor([copy.deepcopy(expert.lamb)], requires_grad=True, dtype=torch.float)
    except:
        temp_lambda = expert.lamb.clone()


    criterion = nn.BCELoss()

    task0 = task[0]
    task1 = task[1]
    task_df = pd.concat([task0, task1])


    X_q, y_q, z_q = sample_data_from_task(task_df, Kq, d_feature)
    X_q = X_q.values
    y_q = y_q.values
    z_q = z_q.values

    X_temp = copy.deepcopy(X_q)
    z_temp = copy.deepcopy(z_q)
    y_temp = copy.deepcopy(y_q)

    X_q = torch.tensor(X_q, dtype=torch.float).unsqueeze(1)
    y_q = torch.tensor(y_q, dtype=torch.float).unsqueeze(1)
    z_q = torch.tensor(z_q, dtype=torch.float).unsqueeze(1)


    y_hat = expert.net.parameterised(X_q, temp_weights)

    loss = criterion(y_hat, y_q)
    fair = (cal_dbc(torch.squeeze(z_q), torch.squeeze(y_hat)) - eps)
    delta_t= delta * expert_eta
    temp_lambda=temp_lambda*0
    loss = (loss / Kq)+temp_lambda*fair - (delta_t / 2) * (temp_lambda ** 2)
    grad_weights = torch.autograd.grad(loss.sum(), temp_weights, retain_graph=True)

    grad_lambda = torch.autograd.grad(loss.sum(), temp_lambda)

    return grad_weights,grad_lambda


