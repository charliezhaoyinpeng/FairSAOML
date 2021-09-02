import numpy as np
from numpy import linalg as LA
import random, math
import pandas as pd


class Expert:
    """
    an object, expert
    """
    def __init__(self, t, length, data, theta, lamb):
        self.t = t
        self.length = length
        self.data = data
        self.R = 0
        self.C = 0
        self.theta = theta
        self.lamb = lamb
        # self.p = random.uniform(0, 1)

    def get_X_by_inds(self, array, d_feature):
        frames = []
        for index in array:
            pos_df = pd.read_csv(datapath + '/task' + str(index) + '/pos.csv')
            neg_df = pd.read_csv(datapath + '/task' + str(index) + '/neg.csv')
            df = pd.concat([pos_df, neg_df])
            frames.append(df)
        new_df = pd.concat(frames)
        X = new_df[new_df.columns[-d_feature:]].copy()
        return X

    def get_eta(self, eps, d_feature):
        arraies = list()
        for i in range(1, self.t + 1):
            ct = get_C_t(i)
            arraies.extend(ct)
        target_arraies = list()
        for array in arraies:
            if len(array) == self.length:
                target_arraies.append(array)
        X_buffer = list()
        for target_array in target_arraies:
            X_buffer.append(self.get_X_by_inds(target_array, d_feature))

        S = math.sqrt(1 + 2 * eps) - 1
        G = max(math.sqrt(d_feature) + S, max(X_buffer))

        self.eta = S / (G * math.sqrt(self.length))


def dgc():
    """
    :param T: total number of tasks
    :return: indices set of all dense geometric coverings
    """
    k = int(np.log2(T))
    res_list = list()
    for i in range(k):
        my_list = np.array(range(T)) + 1
        n = 2 ** i
        ans = [my_list[i * n:(i + 1) * n] for i in range((len(my_list) + n - 1) // n)]
        # print(ans)
        res_list = res_list + ans
    return res_list


def get_C_t(t):
    """
    Given the index (t) of current task, get C_t
    :param T: Total number of tasks
    :param t: the current task index
    :return: C_t, a subset of dgc
    """
    interval_inds = dgc()
    C_t = list()
    for item in interval_inds:
        if item[0] == t:
            C_t.append(item)

    print('C_t: ', C_t)
    return C_t


def get_data_by_Ct(C_t):
    """
    :param datapath: directory of the data set
    :param C_t: a subset of dgc
    :return: a dictionary containing C_t and its corresponding datasets
    """
    ans = {'ct': C_t}
    for indset in C_t:
        key = str(len(indset))
        frames = []
        for index in indset:
            pos_df = pd.read_csv(datapath + '/task' + str(index) + '/pos.csv')
            neg_df = pd.read_csv(datapath + '/task' + str(index) + '/neg.csv')
            df = pd.concat([pos_df, neg_df])
            frames.append(df)
        values = pd.concat(frames)
        ans[key] = values
    # print(ans)
    return ans


if __name__ == "__main__":
    T = 16
    t = 1
    datapath = r"../data/syn_cls"

    ct = get_C_t(t)
    get_data_by_Ct(ct)
