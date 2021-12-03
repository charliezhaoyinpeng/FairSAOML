import time
from os import listdir
from os.path import isfile, join

from cls_pdrftml import *


def running_time(totalsec):
    day = totalsec // (24 * 3600)
    restsec = totalsec % (24 * 3600)
    hour = restsec // 3600
    restsec %= 3600
    minutes = restsec // 60
    restsec %= 60
    seconds = restsec
    print("Total running time: %d days, %d hours, %d minutes, %d seconds." % (day, hour, minutes, seconds))


if __name__ == "__main__":
    # 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000
    lamb = random.choice([0.0001, 0.001, 0.01, 0.1, 1])  # lambda initialization
    val_batch_size = 0.9  # data points for validation
    K = 100  # few shots in support
    Kq = 2 * K  # shots for query
    inner_steps = 1  # gradient steps in the inner loop
    pd_updates = 1  # inner primal-dual iteration
    num_iterations = 100  # outer iteration
    meta_batch = 4

    eta_1 = 0.001  # step size of inner primal update
    eta_2 = 0.001  # step size of inner dual update
    eta_3 = 0.001  # step size of outer primal update
    eta_4 = 0.001  # step size of outer dual update
    xi = 0.015  # parameter of outer regularization
    delta = 50  # some constant for outer augmentation
    eps = 0.05  # fairness threshold
    num_neighbors = 3

    # cls_syn_data:2; adult:16; communities_and_crime:100; bank:16; census_income:36
    d_feature = 16  # feature size of the data set
    data_path = r'C:\Users\Chen Zhao\Dropbox\PDRFTML\data'
    dataset = r'adult'
    save = r'C:\Users\Chen Zhao\Dropbox\PDRFTML\output_new'
    tasks = [x[0] for x in os.walk(data_path + '\\' + dataset)][1:]

    start = time.time()
    print(lamb)
    pdrftml(d_feature, lamb, tasks, data_path, dataset, save,
            K, Kq, val_batch_size, num_neighbors,
            num_iterations, inner_steps, pd_updates, meta_batch,
            eta_1, eta_2, eta_3, eta_4, delta, eps, xi)
    cost_time_in_second = time.time() - start
    running_time(cost_time_in_second)
