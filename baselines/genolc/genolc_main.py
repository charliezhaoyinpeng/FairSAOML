import time
from os import listdir
from os.path import isfile, join

from genolc import *

if __name__ == "__main__":
    # [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    lamb = random.choice([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000])
    K = 100
    val_batch_size = 0.1  # data points for validation

    eps = 0.05
    eta_1 = 0.01
    delta = 50
    num_neighbors = 3

    # cls_syn_data:2; adult:16; communities_and_crime:100; bank:16; census_income:36
    d_feature = 16  # feature size of the data set
    data_path = r'C:\Users\Chen Zhao\Dropbox\PDRFTML\data'
    dataset = r'bank'
    save = r'C:\Users\Chen Zhao\Dropbox\PDRFTML\output_new'
    tasks = [x[0] for x in os.walk(data_path + '\\' + dataset)][1:]

    start = time.time()
    print(lamb)
    genolc(d_feature, lamb, tasks, data_path, dataset, save,
         K, val_batch_size, num_neighbors,
         eta_1, delta, eps)
    cost_time_in_second = time.time() - start
    cost_time = time.strftime("%H:%M:%S", time.gmtime(cost_time_in_second))
    print(cost_time)
