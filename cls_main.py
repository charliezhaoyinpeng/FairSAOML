import time
from os import listdir
from os.path import isfile, join
import argparse
from cls_fairSAOML import  *


def running_time(totalsec):
    day = totalsec // (24 * 3600)
    restsec = totalsec % (24 * 3600)
    hour = restsec // 3600
    restsec %= 3600
    minutes = restsec // 60
    restsec %= 60
    seconds = restsec
    print("Total running time: %d days, %d hours, %d minutes, %d seconds." % (day, hour, minutes, seconds))

def get_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_batch_size", default=0.9, type=float, help="data points for validation")
    parser.add_argument("--K", default=100, type=int, help="few shots in support")
    parser.add_argument("--Kq", default=200, type=int, help="few shots in query")
    parser.add_argument("--inner_steps", default=2, type=int, help="gradient steps in the inner loop")
    parser.add_argument("--pd_updates", default=3, type=int, help="inner primal-dual iteration")
    parser.add_argument("--eta_1", default=0.0001, type=float, help="step size of inner primal update for validation model")
    parser.add_argument("--eta_2", default=0.0001, type=float, help="step size of inner dual update for validation model")
    parser.add_argument("--meta_eta_1", default=0.0001, type=float,
                        help="step size for outer primal update")
    parser.add_argument("--meta_eta_2", default=0.0001, type=float,
                        help="step size for outer dual update")
    parser.add_argument("--num_iterations", default=5, type=int, help="outer iteration")
    parser.add_argument("--xi", default=0.015, type=float, help="parameter of outer regularization")
    parser.add_argument("--delta", default=10, type=int, help="some constant for outer augmentation")
    parser.add_argument("--eps", default=0.05, type=float, help="fairness threshold")
    parser.add_argument("--d_feature", default=2, type=int, help="feature size of the data set. for example, cls_syn_data:2; adult:16; communities_and_crime:100; bank:16; census_income:36")
    parser.add_argument("--data_path", default=r'C:\Users\fengm\Desktop\pdrftml\data\data', type=str, help="root to all dataset")
    parser.add_argument("--dataset", default=r'syn_cls_reverse_medium', type=str, help="dataset folder name")
    parser.add_argument("--save", default=r'C:\Users\fengm\Desktop\pdrftml\output_new', type=str, help="save location")
    parser.add_argument("--meta_batch", default=4,type=int, help="save location")
    parser.add_argument("--radius", default=10, type=int, help="radius")
    parser = parser.parse_args()
    return  parser


if __name__ == "__main__":
    # 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000
    lamb = random.choice([0.0001, 0.001, 0.01, 0.1, 1])  # lambda initialization
    parser = get_parameter()
    val_batch_size = parser.val_batch_size  # data points for validation
    K = parser.K  # few shots in support
    Kq = parser.Kq  # shots for query
    inner_steps = parser.inner_steps  # gradient steps in the inner loop
    pd_updates = parser.pd_updates  # inner primal-dual iteration
    num_iterations = parser.num_iterations  # outer iteration
    meta_batch = parser.meta_batch
    delta = parser.delta

    eta_1 = parser.eta_1  # step size of inner primal update
    eta_2 = parser.eta_1  # step size of inner dual update
    meta_eta_1= parser.meta_eta_1
    meta_eta_2 = parser.meta_eta_2

    xi = parser.xi  # parameter of outer regularization
    # delta = parser.delta  # some constant for outer augmentation
    eps = parser.eps  # fairness threshold
    # eps = 0.35  # new fairness threshold
    num_neighbors = 3

    # cls_syn_data:2; adult:16; communities_and_crime:100; bank:16; census_income:36 Syn_cls 2
    d_feature = parser.d_feature  # feature size of the data set
    data_path = parser.data_path
    dataset = parser.dataset
    radius = parser.radius
    # data_path = r"C:\Users\fengm\Desktop\pdrftml\data\data\syn_cls"
    # dataset = r'communities_and_crime'
    save = parser.save
    tasks = [x[0] for x in os.walk(os.path.join(data_path, dataset))][1:]
    # print("123",tasks)

    start = time.time()
    print(lamb)
    dp,eop,acc,loss=fairSAOML(d_feature, lamb, tasks, data_path, dataset, save,
            K, Kq, val_batch_size, num_neighbors,
            num_iterations, inner_steps, pd_updates, meta_batch,
            eta_1, eta_2, eps, xi,radius,meta_eta_1,meta_eta_2,delta)
    cost_time_in_second = time.time() - start
    running_time(cost_time_in_second)

    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import pandas as pd
    from numpy import genfromtxt

    fig, ax = plt.subplots()

    sns.set(font_scale=0.8)
    clrs = sns.color_palette("husl", 7)
    plt.ylabel("dp_fair" + ' value')
    plt.xlabel('task')
    plt.title("synthetic dataset" + 'domain shift validation')
    # plt.figure(figsize=(cm_to_inch(50),cm_to_inch(40)))
    with sns.axes_style("darkgrid"):
        for i in range(1):
            epochs = list(range(len(tasks)))
            ax.plot(epochs, dp , label="fairSAOML",c=clrs[i])
            ax.legend()
    plt.show()


    fig, ax = plt.subplots()

    sns.set(font_scale=0.8)
    clrs = sns.color_palette("husl", 7)
    plt.ylabel("eop_fair" + ' value')
    plt.xlabel('task')
    plt.title("synthetic dataset" + 'domain shift validation')
    # plt.figure(figsize=(cm_to_inch(50),cm_to_inch(40)))
    with sns.axes_style("darkgrid"):
        for i in range(1):
            epochs = list(range(len(tasks)))
            ax.plot(epochs, eop ,label="fairSAOML", c=clrs[i])
            ax.legend()
    plt.show()

    fig, ax = plt.subplots()

    sns.set(font_scale=0.8)
    clrs = sns.color_palette("husl", 7)
    plt.ylabel("loss_fair" + ' value')
    plt.xlabel('task')
    plt.title("synthetic dataset" + 'domain shift validation')
    # plt.figure(figsize=(cm_to_inch(50),cm_to_inch(40)))
    with sns.axes_style("darkgrid"):
        for i in range(1):
            epochs = list(range(len(tasks)))
            ax.plot(epochs, loss ,label="fairSAOML", c=clrs[i])
            ax.legend()
    plt.show()

# if __name__ == "__main__":
#     # 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000
#     lamb = random.choice([0.0001, 0.001, 0.01, 0.1, 1])  # lambda initialization
#     val_batch_size = 0.9  # data points for validation
#     K = 100  # few shots in support
#     Kq = 2 * K  # shots for query
#     inner_steps = 1  # gradient steps in the inner loop
#     pd_updates = 1  # inner primal-dual iteration
#     num_iterations = 100  # outer iteration
#     meta_batch = 4
#
#     eta_1 = 0.001  # step size of inner primal update
#     eta_2 = 0.001  # step size of inner dual update
#     eta_3 = 0.001  # step size of outer primal update
#     eta_4 = 0.001  # step size of outer dual update
#     xi = 0.015  # parameter of outer regularization
#     delta = 50  # some constant for outer augmentation
#     eps = 0.05  # fairness threshold
#     # eps = 0.35  # new fairness threshold
#     num_neighbors = 3
#
#     # cls_syn_data:2; adult:16; communities_and_crime:100; bank:16; census_income:36
#     d_feature = 100  # feature size of the data set
#     data_path = r'C:\Users\fengm\Desktop\pdrftml\data\data'
#     dataset = r'syn_cls'
#     # data_path = r"C:\Users\fengm\Desktop\pdrftml\data\data\syn_cls"
#     # dataset = r'communities_and_crime'
#     save = r'C:\Users\fengm\Desktop\pdrftml\output_new'
#     tasks = [x[0] for x in os.walk(data_path + '\\' + dataset)][1:]
#     # print("123",tasks)
#
#     start = time.time()
#     print(lamb)
#     pdrftml(d_feature, lamb, tasks, data_path, dataset, save,
#             K, Kq, val_batch_size, num_neighbors,
#             num_iterations, inner_steps, pd_updates, meta_batch,
#             eta_1, eta_2, eta_3, eta_4, delta, eps, xi)
#     cost_time_in_second = time.time() - start
#     running_time(cost_time_in_second)
