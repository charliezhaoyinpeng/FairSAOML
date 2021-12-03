import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd
import os
import numpy as np


def draw(directory):
    # directory = '/home/mifeng/pdrftml/output_new/2021-10-18-21-43-50-Ours-syn_cls_reverse_medium2'

    np_file = os.path.join(directory, 'val.np')

    data = np.load(np_file, allow_pickle=True)

    dp = data['dp']

    eop = data['eop']

    acc = data['acc']

    loss = data['loss']

    dbc = data['dbc']
    dbc_loss=data['dbc_loss']

    accumulative_loss = []
    accumulative = 0
    for each in loss:
        accumulative += each
        accumulative_loss.append(accumulative)
    accumulative_loss

    accumulative_loss_plus_dbc=[]
    accumulative = 0
    for each in dbc_loss:
        accumulative += each
        accumulative_loss_plus_dbc.append(accumulative)
    accumulative_loss_plus_dbc

    experts_data = data['experts_data']

    number_of_experts = len(experts_data[0][1])

    number_of_tasks = len(experts_data)

    P_for_all_experts = []

    time = 1
    for each_time in experts_data:
        experts = each_time[time]
        #     print(experts)
        if time == 1:
            for each_expert in experts:
                cur_expert = []
                p = each_expert[' p: ']
                cur_expert.append(p)
                P_for_all_experts.append(cur_expert)
        else:
            for index in range(len(experts)):
                p = experts[index][' p: ']
                P_for_all_experts[index].append(p)
        time += 1

    # In[20]:

    save_dir = os.path.join(directory, 'dp_fair.png')
    fig, ax = plt.subplots()
    sns.set(font_scale=0.8)
    clrs = sns.color_palette("husl", 7)
    plt.ylabel("dp_fair" + ' value')
    plt.xlabel('task')
    plt.title("synthetic dataset" + 'domain shift validation')
    # plt.xlim([25, 50])
    # plt.ylim([25, 50])
    # plt.figure(figsize=(cm_to_inch(50),cm_to_inch(40)))
    with sns.axes_style("darkgrid"):
        for i in range(1):
            epochs = list(range(number_of_tasks))
            ax.plot(epochs, dp, label="fairSAOML", c=clrs[i])
            ax.legend()
        plt.savefig(save_dir)

    # In[21]:

    save_dir = os.path.join(directory, 'eop_fair.png')
    fig, ax = plt.subplots()
    sns.set(font_scale=0.8)
    clrs = sns.color_palette("husl", 7)
    plt.ylabel("eop_fair" + ' value')
    plt.xlabel('task')
    plt.title("synthetic dataset" + 'domain shift validation')
    # plt.figure(figsize=(cm_to_inch(50),cm_to_inch(40)))
    with sns.axes_style("darkgrid"):
        for i in range(1):
            epochs = list(range(number_of_tasks))
            ax.plot(epochs, eop, label="fairSAOML", c=clrs[i])
            ax.legend()
        plt.savefig(save_dir)

    # In[22]:

    save_dir = os.path.join(directory, 'acc.png')
    fig, ax = plt.subplots()
    sns.set(font_scale=0.8)
    clrs = sns.color_palette("husl", 7)
    plt.ylabel("acc" + ' value')
    plt.xlabel('task')
    plt.title("synthetic dataset" + 'domain shift validation')
    # plt.figure(figsize=(cm_to_inch(50),cm_to_inch(40)))
    with sns.axes_style("darkgrid"):
        for i in range(1):
            epochs = list(range(number_of_tasks))
            ax.plot(epochs, acc, label="fairSAOML", c=clrs[i])
            ax.legend()
        plt.savefig(save_dir)

    # In[28]:

    save_dir = os.path.join(directory, 'loss.png')
    fig, ax = plt.subplots()
    sns.set(font_scale=0.8)
    clrs = sns.color_palette("husl", 7)
    plt.ylabel("loss" + ' value')
    plt.xlabel('task')
    plt.title("synthetic dataset" + 'domain shift validation')
    # plt.figure(figsize=(cm_to_inch(50),cm_to_inch(40)))
    with sns.axes_style("darkgrid"):
        for i in range(1):
            epochs = list(range(number_of_tasks))[1:]
            ax.plot(epochs, loss[1:], label="fairSAOML", c=clrs[i])
            ax.legend()
        plt.savefig(save_dir)

    # In[24]:

    save_dir = os.path.join(directory, 'accumulative_loss.png')
    fig, ax = plt.subplots()
    sns.set(font_scale=0.8)
    clrs = sns.color_palette("husl", 7)
    plt.ylabel("accumulative_loss" + ' value')
    plt.xlabel('task')
    plt.title("synthetic dataset" + 'domain shift validation')
    # plt.figure(figsize=(cm_to_inch(50),cm_to_inch(40)))
    with sns.axes_style("darkgrid"):
        for i in range(1):
            epochs = list(range(number_of_tasks))
            ax.plot(epochs, accumulative_loss, label="fairSAOML", c=clrs[i])
            ax.legend()
        plt.savefig(save_dir)

    accumulative_loss_plus_dbc

    save_dir = os.path.join(directory, 'accumulative_loss_plus_dbc.png')
    fig, ax = plt.subplots()
    sns.set(font_scale=0.8)
    clrs = sns.color_palette("husl", 7)
    plt.ylabel("accumulative_loss_plus_dbc" + ' value')
    plt.xlabel('task')
    plt.title("synthetic dataset" + 'domain shift validation')
    # plt.figure(figsize=(cm_to_inch(50),cm_to_inch(40)))
    with sns.axes_style("darkgrid"):
        for i in range(1):
            epochs = list(range(number_of_tasks))
            ax.plot(epochs, accumulative_loss_plus_dbc, label="fairSAOML", c=clrs[i])
            ax.legend()
        plt.savefig(save_dir)
    # In[25]:

    labels = []
    index = 1
    for i in range(len(P_for_all_experts)):
        label = 'expert_' + str(index)
        index = index + 1;
        labels.append(label)

    # In[26]:

    labels

    # In[27]:

    save_dir = os.path.join(directory, 'Ps_for_expert.png')
    fig, ax = plt.subplots()
    sns.set(font_scale=0.8)
    fig.set_size_inches(18.5, 10.5)
    clrs = sns.color_palette("husl", 9)
    plt.ylabel("Ps for expert" + ' value')
    plt.xlabel('task')
    plt.title("synthetic dataset" + 'domain shift validation')
    # plt.figure(figsize=(cm_to_inch(50),cm_to_inch(40)))
    with sns.axes_style("darkgrid"):
        for i in range(len(P_for_all_experts)):
            epochs = list(range(number_of_tasks))[1:]
            ax.plot(epochs, P_for_all_experts[i][1:], label=labels[i], c=clrs[i])
            ax.legend()
        plt.savefig(save_dir)


    save_dir = os.path.join(directory, 'dbc.png')
    fig, ax = plt.subplots()
    sns.set(font_scale=0.8)
    clrs = sns.color_palette("husl", 7)
    plt.ylabel("dbc" + ' value')
    plt.xlabel('task')
    plt.title("synthetic dataset" + 'domain shift validation')
    # plt.figure(figsize=(cm_to_inch(50),cm_to_inch(40)))
    with sns.axes_style("darkgrid"):
        for i in range(1):
            epochs = list(range(number_of_tasks))
            ax.plot(epochs, dbc, label="fairSAOML", c=clrs[i])
            ax.legend()
        plt.savefig(save_dir)

    save_dir = os.path.join(directory, 'dbc_plus_loss.png')
    fig, ax = plt.subplots()
    sns.set(font_scale=0.8)
    clrs = sns.color_palette("husl", 7)
    plt.ylabel("dbc_plus_loss" + ' value')
    plt.xlabel('task')
    plt.title("synthetic dataset" + 'domain shift validation')
    # plt.figure(figsize=(cm_to_inch(50),cm_to_inch(40)))
    with sns.axes_style("darkgrid"):
        for i in range(1):
            epochs = list(range(number_of_tasks))
            ax.plot(epochs, dbc_loss, label="fairSAOML", c=clrs[i])
            ax.legend()
        plt.savefig(save_dir)