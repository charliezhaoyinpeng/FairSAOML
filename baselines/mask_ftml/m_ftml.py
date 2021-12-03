from __future__ import division
import torch, os
import torch.nn as nn
import random
import math
import copy, time, pickle, datetime
import pandas as pd
import numpy as np
from numpy import linalg as LA

from m_ftml_eval_metrics_cls import *
from m_ftml_NN import *
from m_ftml_single_task import *
from m_ftml_params_table import *

seed = 34
np.random.seed(seed)
torch.manual_seed(seed)

def meta_update(d_feature, net,
                K, Kq, num_neighbors,
                num_iterations, buffer, inner_steps, meta_batch,
                eta_1, eta_3):

    weights = list(net.parameters())

    if len(buffer) <= meta_batch:
        batch = copy.deepcopy(buffer)
    else:
        batch = random.sample(buffer, meta_batch)

    # batch = copy.deepcopy(buffer)

    for iter in range(1, num_iterations + 1):
        meta_loss = 0
        for i in range(len(batch)):
            task = batch[i]
            t_loss, t_fair, t_acc, t_dp, t_eop, t_disc, t_cons = cal_loss_and_fairness(d_feature, net, task,
                                                                                       K, Kq, num_neighbors,
                                                                                       inner_steps,
                                                                                       eta_1)
            meta_loss += t_loss

        meta_grads = torch.autograd.grad(meta_loss, weights)
        temp_weights = [w.clone() for w in weights]
        weights = [w - eta_3 * g for w, g in zip(temp_weights, meta_grads)]

        weights = list(nn.parameter.Parameter(item) for item in weights)

        net.assign(weights)

    return net


def prep(save, d_feature, dataset,
         K, Kq, val_batch_size, num_neighbors,
         num_iterations, inner_steps, meta_batch,
         eta_1, eta_3):
    now = datetime.datetime.now()
    exp_name = now.strftime("\%Y-%m-%d-%H-%M-%S-Masked-FTML-" + dataset)
    save_folder = save + exp_name
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # model_save_path = save_folder + r"\model.pth"
    val_save_path = save_folder + r"\val.txt"
    with open(save_folder + r'\hyper-parameters.txt', 'wb') as f:
        t = hyper_params(d_feature, dataset,
                         K, Kq, val_batch_size, num_neighbors,
                         num_iterations, inner_steps, meta_batch,
                         eta_1, eta_3)
        f.write(t)

    return val_save_path


def mftml(d_feature, tasks, data_path, dataset, save,
          K, Kq, val_batch_size, num_neighbors,
          num_iterations, inner_steps, meta_batch,
          eta_1, eta_3):
    val_save_path = prep(save, d_feature, dataset,
                         K, Kq, val_batch_size, num_neighbors,
                         num_iterations, inner_steps, meta_batch,
                         eta_1, eta_3)
    net = NN(d_feature)
    buffer = []
    T = len(tasks)
    res = []

    for t in range(1, T + 1):
        start_time = time.time()
        task0 = pd.read_csv(data_path + '\\' + dataset + r'/task' + str(t) + r'/task' + str(t) + '_neg.csv')
        task1 = pd.read_csv(data_path + '\\' + dataset + r'/task' + str(t) + r'/task' + str(t) + '_pos.csv')
        task = [task0, task1]
        buffer.append(task)

        new_net = meta_update(d_feature, net,
                              K, Kq, num_neighbors,
                              num_iterations, buffer, inner_steps, meta_batch,
                              eta_1, eta_3)

        loss_val, fair_val, accuracy_val, dp_val, eop_val, discrimination_val, consistency_val = cal_loss_and_fairness(d_feature, new_net, task,
                                                                                                                       K, val_batch_size, num_neighbors,
                                                                                                                       inner_steps,
                                                                                                                       eta_1)
        cost_time = time.time() - start_time
        print("Val-Task %s/%s: loss:%s; dbc:%s; acc:%s ;dp:%s; eop:%s; disc:%s; cons:%s; time:%s sec." % (
            t, T, np.round(loss_val.item(), 4), np.round(fair_val, 10), np.round(accuracy_val, 10), np.round(dp_val, 10), np.round(eop_val, 10),
            np.round(discrimination_val, 10), np.round(consistency_val, 10),
            np.round(cost_time, 4)))
        # torch.save(net.state_dict(), model_save_path)
        res.append([loss_val.item(), fair_val, accuracy_val, dp_val, eop_val, discrimination_val, consistency_val, cost_time])
        net = copy.deepcopy(new_net)

    with open(val_save_path, 'wb') as f:
        pickle.dump(res, f)
