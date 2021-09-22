from __future__ import division
import torch, os
import torch.nn as nn
import random
import mathFfF
import copy, time, pickle, datetime
import pandas as pd
import numpy as np
from numpy import linalg as LA
from sklearn.metrics import accuracy_score

from eval_metrics_cls import *
from cls_NN import *


def sample_data_from_task(task, batch_size, d_feature):
    data = task.sample(batch_size)
    X = data[data.columns[-d_feature:]].copy()
    y = data[["y"]]
    z = data[["z"]]
    return X, y, z


def mean(a):
    return sum(a).to(dtype=torch.float) / len(a)


def cal_ft(net,z,y,y_hat,K,eps,temp_lambda):
    fair = cal_dbc(torch.squeeze(z), torch.squeeze(y)) - eps
    ones = torch.tensor(np.ones((len(y), 1)), dtype=torch.float).unsqueeze(1)
    loss = (-1.0) * torch.mean(y * torch.log(y_hat) + (ones - y) * torch.log(ones - y_hat)) + temp_lambda * fair
    loss = loss / K
    return loss

def cal_meta_eta_and_reg(S,G,t):
    beta = random.random()
    eta1 = S/(G*math.pow(t, beta))
    eta2 = math.pow(t, beta)/(6*S*G*(t+1))
    delta_t = (6*S*G)/math.pow(t, beta)
    return eta1,eta2,delta_t

def validate_performance(t, d_feature, net, lamb, task,
                          K, val_batch_size, num_neighbors,
                          inner_steps, pd_updates,
                          eta_1, eta_2, eps, xi):
    temp_weights = [w.clone() for w in list(net.parameters())]
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
    # print("X_s",X_s)

    for co_update in range(pd_updates):
        for step in range(inner_steps):
            y_hat = net.parameterised(X_s, temp_weights)
            fair = cal_dbc(torch.squeeze(z_s),torch.squeeze(y_hat))-eps
            # print(fair)
            # fair = torch.abs(torch.mean((z_s - z_bar) * y_hat)) - eps
            # print("fair is",fair,"fair new",fair_new)
            loss = (-1.0) * torch.mean(y_s * torch.log(y_hat) + (ones - y_s) * torch.log(ones - y_hat)) + temp_lambda * fair
            loss = loss / K
            grad = torch.autograd.grad(loss.sum(), temp_weights, retain_graph=True)
            temp_weights = [w - eta_1 * g for w, g in zip(temp_weights, grad)]
            # temp_weights_norm = net.e_norm(temp_weights)
            # if temp_weights_norm > 1:
            #     temp_weights = [w / net.e_norm(temp_weights) for w in temp_weights]

            new_y_hat = net.parameterised(X_s, temp_weights)
            # fair = torch.abs(torch.mean((z_s - z_bar) * new_y_hat)) - eps
            fair = cal_dbc(torch.squeeze(z_s), torch.squeeze(y_hat)) - eps
            loss = (-1.0) * torch.mean(y_s * torch.log(new_y_hat) + (ones - y_s) * torch.log(ones - new_y_hat)) + temp_lambda * fair
            loss = loss / K
            # print(temp_lambda)
            grad_lamb = torch.autograd.grad(loss.sum(), temp_lambda)
            # print(grad_lamb, type(grad_lamb), grad_lamb[0])
            # print(eta_2 * grad_lamb)
            temp_lambda = temp_lambda + eta_2 * grad_lamb[0]
            if temp_lambda.item() < 0:
                temp_lambda = torch.tensor([0], requires_grad=True, dtype=torch.float)

    # if val_batch_size < 1: # means for validation purpose
    Kq = round(len(task_df.index) * val_batch_size)

    X_q, y_q, z_q = sample_data_from_task(task_df, Kq, d_feature)
    X_q = X_q.values
    y_q = y_q.values
    z_q = z_q.values

    X_temp = copy.deepcopy(X_q)
    z_temp = copy.deepcopy(z_q)
    y_temp = copy.deepcopy(y_q)
    z_bar = np.mean(z_q) * np.ones((len(z_q), 1))

    X_q = torch.tensor(X_q, dtype=torch.float).unsqueeze(1)
    y_q = torch.tensor(y_q, dtype=torch.float).unsqueeze(1)
    z_q = torch.tensor(z_q, dtype=torch.float).unsqueeze(1)
    z_bar = torch.tensor(z_bar, dtype=torch.float).unsqueeze(1)

    y_hat = net.parameterised(X_q, temp_weights)
    loss = criterion(y_hat, y_q)
    loss = loss / Kq

    fair = torch.abs(torch.mean((z_q - z_bar) * y_hat)).item()

    y_hat = y_hat.detach().numpy().reshape(len(y_hat), 1)
    y_q = y_q.detach().numpy().reshape(len(y_q), 1)

    input_zy = np.column_stack((z_temp, y_hat))
    z_y_hat_y = np.column_stack((input_zy, y_temp))
    yX = np.column_stack((y_hat, X_temp))

    accuracy = accuracy_score(y_hat.round(), y_q)
    dp = cal_dp(input_zy, t-1, xi)
    eop = cal_eop(z_y_hat_y, t-1, xi)
    discrimination = cal_discrimination(input_zy)*100
    # consistency = cal_consistency(yX, num_neighbors)
    consistency = 1

    return loss, fair, accuracy, dp, eop, discrimination, consistency





def cal_loss_and_fairness(t, d_feature, net, lamb, task,
                          K, Kq, num_neighbors,
                          inner_steps, pd_updates,
                          eta_1, eta_2, eps, xi):
    temp_weights = [w.clone() for w in list(net.parameters())]
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
    # print("X_s",X_s)

    for co_update in range(pd_updates):
        for step in range(inner_steps):
            y_hat = net.parameterised(X_s, temp_weights)
            fair = cal_dbc(torch.squeeze(z_s),torch.squeeze(y_hat))-eps
            # print(fair)
            # fair = torch.abs(torch.mean((z_s - z_bar) * y_hat)) - eps
            # print("fair is",fair,"fair new",fair_new)
            loss = (-1.0) * torch.mean(y_s * torch.log(y_hat) + (ones - y_s) * torch.log(ones - y_hat)) + temp_lambda * fair
            loss = loss / K
            grad = torch.autograd.grad(loss.sum(), temp_weights, retain_graph=True)
            temp_weights = [w - eta_1 * g for w, g in zip(temp_weights, grad)]
            # temp_weights_norm = net.e_norm(temp_weights)
            # if temp_weights_norm > 1:
            #     temp_weights = [w / net.e_norm(temp_weights) for w in temp_weights]

            new_y_hat = net.parameterised(X_s, temp_weights)
            # fair = torch.abs(torch.mean((z_s - z_bar) * new_y_hat)) - eps
            fair = cal_dbc(torch.squeeze(z_s), torch.squeeze(y_hat)) - eps
            loss = (-1.0) * torch.mean(y_s * torch.log(new_y_hat) + (ones - y_s) * torch.log(ones - new_y_hat)) + temp_lambda * fair
            loss = loss / K
            # print(temp_lambda)
            grad_lamb = torch.autograd.grad(loss.sum(), temp_lambda)
            # print(grad_lamb, type(grad_lamb), grad_lamb[0])
            # print(eta_2 * grad_lamb)
            temp_lambda = temp_lambda + eta_2 * grad_lamb[0]
            if temp_lambda.item() < 0:
                temp_lambda = torch.tensor([0], requires_grad=True, dtype=torch.float)

    if Kq < 1: # means for validation purpose
        Kq = round(len(task_df.index) * Kq)

    X_q, y_q, z_q = sample_data_from_task(task_df, Kq, d_feature)
    X_q = X_q.values
    y_q = y_q.values
    z_q = z_q.values

    X_temp = copy.deepcopy(X_q)
    z_temp = copy.deepcopy(z_q)
    y_temp = copy.deepcopy(y_q)
    z_bar = np.mean(z_q) * np.ones((len(z_q), 1))

    X_q = torch.tensor(X_q, dtype=torch.float).unsqueeze(1)
    y_q = torch.tensor(y_q, dtype=torch.float).unsqueeze(1)
    z_q = torch.tensor(z_q, dtype=torch.float).unsqueeze(1)
    z_bar = torch.tensor(z_bar, dtype=torch.float).unsqueeze(1)

    y_hat = net.parameterised(X_q, temp_weights)
    loss = criterion(y_hat, y_q)
    loss = loss / Kq

    fair = torch.abs(torch.mean((z_q - z_bar) * y_hat)).item()

    y_hat = y_hat.detach().numpy().reshape(len(y_hat), 1)
    y_q = y_q.detach().numpy().reshape(len(y_q), 1)

    input_zy = np.column_stack((z_temp, y_hat))
    z_y_hat_y = np.column_stack((input_zy, y_temp))
    yX = np.column_stack((y_hat, X_temp))

    accuracy = accuracy_score(y_hat.round(), y_q)
    dp = cal_dp(input_zy, t-1, xi)
    eop = cal_eop(z_y_hat_y, t-1, xi)
    discrimination = cal_discrimination(input_zy)*100
    # consistency = cal_consistency(yX, num_neighbors)
    consistency = 1

    return loss, fair, accuracy, dp, eop, discrimination, consistency
