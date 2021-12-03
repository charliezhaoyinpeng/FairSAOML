import time
from os import listdir
from os.path import isfile, join

from m_ftml import *

if __name__ == "__main__":
    val_batch_size = 0.9  # data points for validation
    K = 100  # few shots in support
    Kq = 2 * K  # shots for query
    inner_steps = 1  # gradient steps in the inner loop
    num_iterations = 10  # outer iteration
    meta_batch = 32

    eta_1 = 0.01  # step size of inner primal update
    eta_3 = 0.01  # step size of outer primal update
    num_neighbors = 3

    # cls_syn_data:2; adult:16; communities_and_crime:100; bank:16; census_income:36
    d_feature = 16  # feature size of the data set
    data_path = r'C:\Users\Chen Zhao\Dropbox\PDRFTML\data'
    dataset = r'bank'
    save = r'C:\Users\Chen Zhao\Dropbox\PDRFTML\output_new'
    tasks = [x[0] for x in os.walk(data_path + '\\' + dataset)][1:]

    start = time.time()
    mftml(d_feature, tasks, data_path, dataset, save,
         K, Kq, val_batch_size, num_neighbors,
         num_iterations, inner_steps, meta_batch,
         eta_1, eta_3)
    cost_time_in_second = time.time() - start
    cost_time = time.strftime("%H:%M:%S", time.gmtime(cost_time_in_second))
    print(cost_time)
