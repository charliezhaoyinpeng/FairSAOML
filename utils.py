import numpy as np
from numpy import linalg as LA
import random, math
import pandas as pd
import copy
global T
global datapath
import  torch
from numpy import linalg as LA
import os

def initialize_set_U(task_length, data_path, lamb, net, eps, d_feature):
    """
    Based on the task_length, we calculate the number of experts we need, and assign them the initial paramenter and data.
    """
    global T
    T = task_length
    global datapath
    datapath = data_path
    total_experts, ct = get_details_for_total_experts();
    # print("total experts", total_experts,"length",ct)
    data = get_data_by_Ct(ct)

    U = []
    for i in range(total_experts):
        length = len(ct[i])
        cur_data = data[str(length)]
        p = 1
        lamb = lamb
        net = net
        # print("i",i,"cur_data",cur_data,'len',length)
        expert = Expert(t=1, length=length, data=cur_data, net=net, lamb=lamb, p=p,eps=eps,d_feature=d_feature)
        expert.update_S_G_eta()
        U.append(expert)
    # print("total experts", total_experts, "length", ct)
    #
    # print("the initialize set A is",A)

    return U


def update_experts_at_t(A, t, lamb, net):
    """
    update data assigned to actve eperts and give the meta net parameter and meta lambda to active experts
    """
    ct = get_C_t(t)
    data = get_data_by_Ct(ct)
    active_number = len(
        ct)  # If In is active at time t, In-1 must active.So we only need to know how many the length l in the ct set.
    print("@@@@@@current time",t,"active_number",active_number)

    for i in range(active_number):
        cur_expert = A[i]
        length = len(ct[i])
        if (t-1)%length ==0:  # means the current expert at the start of CI
            cur_expert.R =0
            cur_expert.C =0
        update_data = data[str(length)]
        cur_expert.t = t
        cur_expert.data = update_data
        # temp_weights = [w.clone() for w in list(net.parameters())]
        # cur_expert.net = net
        temp_weights = [w.clone() for w in list(net.parameters())]
        cur_expert.weights = temp_weights
        cur_expert.lamb = lamb
        cur_expert.update_S_G_eta()

    return A, active_number


class Expert:
    """
    an object, expert
    """

    def __init__(self, t, length, data, net, lamb, p,eps,d_feature):
        self.t = t  # current time
        self.length = length
        self.data = data
        self.R = 0
        self.C = 0
        # self.theta = theta
        self.net = net  # theta
        temp_weights = [w.clone() for w in list(net.parameters())]
        self.weights =temp_weights
        self.lamb = lamb
        self.p = p
        # self.ft = None
        self.eps=eps
        self.d_feature=d_feature
        self.w =0

    def print_expert(self):
        print("expert CI length: ", self.length, " R: ", float(self.R), " C: ", float(self.C), " lamb: ", float(self.lamb), " p: ",float(self.p),"w:",float(self.w))
        return {"expert CI length: ": self.length, " R: ": float(self.R), " C: ": float(self.C), " lamb: ": float(self.lamb), " p: ":float(self.p),"w":float(self.w)}

    def get_X_by_inds(self, array, d_feature):
        frames = []
        for index in array:
            pos_df = pd.read_csv(os.path.join(datapath ,'task'+str(index) , 'pos.csv'))
            neg_df = pd.read_csv(os.path.join(datapath , 'task'+ str(index) ,'neg.csv'))
            df = pd.concat([pos_df, neg_df])
            frames.append(df)
        new_df = pd.concat(frames)
        X = new_df[new_df.columns[-d_feature:]].copy()
        return X  ## non protective atribute

    def update_S_G_eta(self):
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
            X_buffer.append(self.get_X_by_inds(target_array, self.d_feature))
        # print("xbuffer",X_buffer)
        # print("lenxbuffuer",len(X_buffer))
        max_e_i = -1;
        for each in X_buffer:
            max_e_i = max(max_e_i,LA.norm(each))
        S = math.sqrt(1 + 2 * self.eps) - 1
        # G = max(math.sqrt(d_feature) + S, max(X_buffer))
        G = max(math.sqrt(self.d_feature + 1) + S, max_e_i)
        self.S = S
        self.G = G
        self.eta = S / (G * math.sqrt(self.length))  ###### update for each time t


def dgc():
    """
    :param T: total number of tasks
    :return: indices set of all dense geometric coverings
    """
    global T
    # print(T)
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
    # print("interval_inds",interval_inds)
    C_t = list()
    for item in interval_inds:
        if item[0] == t:
            C_t.append(item)

    # print('C_t: ', C_t)
    return C_t


def get_details_for_total_experts():
    """
    Consider the number of tasks, found the total experts and initial task length needed for anlysising the dataset.
    """
    return len(get_C_t(t=1)), get_C_t(t=1)


def get_data_by_Ct(C_t):
    """
    :param datapath: directory of the data set
    :param C_t: a subset of dgc
    :return: a dictionary containing C_t and its corresponding datasets
    """
    ans = {'ct': C_t}
    for indset in C_t:
        key = str(len(indset))
        # print("now key",key)
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
    T = 512
    t = 257
    datapath = r"C:\Users\fengm\Desktop\pdrftml\data\data\syn_cls"

    # dgc()
    #
    ct = get_C_t(t)
    print(get_C_t(t))
    print(len(get_C_t(t)))
    # print("ct", ct)
    # print("get_data_by_Ct(ct)", get_data_by_Ct(ct))
    print("dgc", dgc())
    # initialize_set_A()
