import numpy as np
from numpy import linalg as LA
import random, math
import pandas as pd

global T
global datapath


def initialize_set_A(task_length,data_path,lamb,net):
    global T
    T= task_length
    global datapath
    datapath = data_path
    total_experts,ct=get_details_for_total_experts();
    print("total experts", total_experts,"length",ct)
    data = get_data_by_Ct(ct)

    A=[]
    for i in range(total_experts):
        length=len(ct[i])
        cur_data =data[str(length)]
        p=1
        lamb =lamb
        net = net
        print("i",i,"cur_data",cur_data,'len',length)
        expert = Expert(t=1,length=length,data=cur_data,net=net,lamb = lamb,p=p)
        A.append(expert)
    # print("total experts", total_experts, "length", ct)
    #
    # print("the initialize set A is",A)

    return A
def update_data_in_A_at_t(A,t,lamb,net):
    ct = get_C_t(t)
    data = get_data_by_Ct(ct)
    active_number = len(ct) # If In is active at time t, In-1 must active.So we only need to know how many the length l in the ct set.

    for i in range(ct):
        cur_expert = A[i]
        length = len(ct[i])
        update_data = data[str(length)]
        cur_expert.data = update_data
        cur_expert.net = net
        cur_expert.lamb = lamb

    return A,active_number


class Expert:
    """
    an object, expert
    """
    def __init__(self, t, length, data, net, lamb,p):
        self.t = t  # current time
        self.length = length
        self.data = data
        self.R = 0
        self.C = 0
        # self.theta = theta
        self.net = net # theta
        self.lamb = lamb
        self.p = p
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
        return X ## non protective atribute

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
        print("xbuffer",X_buffer)

        S = math.sqrt(1 + 2 * eps) - 1
        G = max(math.sqrt(d_feature) + S, max(X_buffer))
        self.S = S
        self.G = G
        self.eta = S / (G * math.sqrt(self.length))


def dgc():
    """
    :param T: total number of tasks
    :return: indices set of all dense geometric coverings
    """
    global T
    print(T)
    k = int(np.log2(T))
    res_list = list()
    for i in range(k):
        my_list = np.array(range(T)) + 1
        n = 2 ** i
        ans = [my_list[i * n:(i + 1) * n] for i in range((len(my_list) + n - 1) // n)]
        # print(ans)
        res_list = res_list + ans
    # print(res_list)
    return res_list


def get_C_t(t):
    """
    Given the index (t) of current task, get C_t
    :param T: Total number of tasks
    :param t: the current task index
    :return: C_t, a subset of dgc
    """
    interval_inds = dgc()
    print("interval_inds",interval_inds)
    C_t = list()
    for item in interval_inds:
        if item[0] == t:
            C_t.append(item)

    print('C_t: ', C_t)
    return C_t
def get_details_for_total_experts():
    """
    Consider the number of tasks, found the total experts and initial task length needed for anlysising the dataset.
    """
    return len(get_C_t(t=1)),get_C_t(t=1)

def get_data_by_Ct(C_t):
    """
    :param datapath: directory of the data set
    :param C_t: a subset of dgc
    :return: a dictionary containing C_t and its corresponding datasets
    """
    ans = {'ct': C_t}
    for indset in C_t:
        key = str(len(indset))
        print("now key",key)
        frames = []
        for index in indset:
            pos_df = pd.read_csv(datapath + '/task' + str(index) + '/pos.csv')
            neg_df = pd.read_csv(datapath + '/task' + str(index) + '/neg.csv')
            df = pd.concat([pos_df, neg_df])
            frames.append(df)
        values = pd.concat(frames)
        ans[key] = values
    print(ans)
    return ans


if __name__ == "__main__":
    T = 17
    t = 1
    datapath = r"C:\Users\fengm\Desktop\pdrftml\data\data\syn_cls"

    # dgc()
    #
    ct = get_C_t(t)
    get_data_by_Ct(ct)
    # print(dgc())
    # initialize_set_A()
