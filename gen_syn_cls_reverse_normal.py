import math, os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import seed, shuffle
from scipy.stats import multivariate_normal

SEED = 1122334455
seed(SEED)  # set the random seed so that the random permutations can be reproduced again
np.random.seed(SEED)


def gen_gaussian(mean_in, cov_in, class_label, n):
    """
    :param mean_in: Mean of the distribution
    :param cov_in: Covariance matrix
    :param class_label: 0 or 1
    :param n: number of samples
    :return: multivariate_gaussian (nv), feature_vector (X), label_vector (y)
    """
    nv = multivariate_normal(mean=mean_in, cov=cov_in)
    X = nv.rvs(n)
    y = np.ones(n, dtype=float) * class_label

    return nv, X, y


def gen_gaussian_task(mu1, sigma1, mu2, sigma2, disc_factor):
    """
    :param disc_factor: this factor controls the correlation between z and y
    :return: data feature X, data labels y, sensitive variables z
    """
    nv1, X1, y1 = gen_gaussian(mu1, sigma1, 1, int(n_samples / 2))  # positive class
    nv2, X2, y2 = gen_gaussian(mu2, sigma2, 0, int(n_samples / 2))  # negative class

    # join the posisitve and negative class clusters
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))

    # shuffle the data
    perm = list(range(0, n_samples))
    shuffle(perm)
    X = X[perm]
    y = y[perm]

    rotation_mult = np.array(
        [[math.cos(disc_factor), -math.sin(disc_factor)], [math.sin(disc_factor), math.cos(disc_factor)]])
    X_aux = np.dot(X, rotation_mult)

    z = []  # this array holds the sensitive feature value
    for i in range(0, len(X)):
        x = X_aux[i]
        # probability for each cluster that the point belongs to it
        p1 = nv1.pdf(x)
        p2 = nv2.pdf(x)
        # normalize the probabilities from 0 to 1
        s = p1 + p2
        p1 = p1 / s
        p2 = p2 / s
        r = np.random.uniform()  # generate a random number from 0 to 1
        if r < p1:  # the first cluster is the positive class
            z.append(1.0)  # 1.0 means its male
        else:
            z.append(0.0)  # 0.0 means its female
    z = np.array(z)

    return X, y, z


def data_noraml_save(X, y, z, task_ind,shift):
    """
    :param X: data feature matrix
    :param y: data labels
    :param z: sensitive variables
    :param task_ind: task index
    """
    title = ["z", "y", "x1", "x2"]
    zy = np.column_stack((z, y))
    task = np.column_stack((zy, X))
    task = pd.DataFrame(task, columns=title)

    task_norm = (task - task.mean()) / task.std()
    if shift:
        print(task_norm)
        task_norm["x1"] = a*task_norm["x1"]+b
        task_norm["x2"] = a*task_norm["x2"]+b
        print(task_norm)

    task_norm['z'] = task['z']
    task_norm['y'] = task['y']

    task_norm_0 = task_norm[task_norm['y'] == 0]
    task_norm_1 = task_norm[task_norm['y'] == 1]



    # print(task_norm)
    # task_array = task_norm.values
    # print(task_array)
    # X = task_array[:,-2:]
    # z = task_array[:,0]
    # y = task_array[:,1]
    # print("x",X)
    # print(y)
    # print(z)
    # plot_data(X, y, z, 200, "Visualization of the Data Distribution")


    save_folder = save + '/task' + str(task_ind + 1)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if task_norm_0.isnull().values.any():
        print("task_neg %s contains missing values." % (task_ind + 1))
    task_norm_0.to_csv(save_folder + '/neg.csv', index=False)
    if task_norm_1.isnull().values.any():
        print("task_pos %s contains missing values." % (task_ind + 1))
    task_norm_1.to_csv(save_folder + '/pos.csv', index=False)


def plot_data(X, y, z, num_to_draw, title):
    x_draw = X[:num_to_draw]
    y_draw = y[:num_to_draw]
    z_draw = z[:num_to_draw]

    X_z_0 = x_draw[z_draw == 0.0]
    X_z_1 = x_draw[z_draw == 1.0]
    y_z_0 = y_draw[z_draw == 0.0]
    y_z_1 = y_draw[z_draw == 1.0]

    plt.scatter(X_z_0[y_z_0 == 1.0][:, 0], X_z_0[y_z_0 == 1.0][:, 1], color='green', marker='x', s=2, linewidth=1.5,
                label="ProtVar/+Pos")
    plt.scatter(X_z_0[y_z_0 == 0.0][:, 0], X_z_0[y_z_0 == 0.0][:, 1], color='red', marker='x', s=2, linewidth=1.5,
                label="ProtVar/-Neg")
    plt.scatter(X_z_1[y_z_1 == 1.0][:, 0], X_z_1[y_z_1 == 1.0][:, 1], color='green', marker='o', facecolors='none',
                s=30, label="Non-ProtVar/+Pos")
    plt.scatter(X_z_1[y_z_1 == 0.0][:, 0], X_z_1[y_z_1 == 0.0][:, 1], color='red', marker='o', facecolors='none', s=30,
                label="Non-ProtVar/-Neg")

    plt.tick_params(axis='x', which='both', bottom='off', top='off',
                    labelbottom='off')  # dont need the ticks to see the data distribution
    plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
    plt.legend(loc=2, fontsize=12)
    plt.xlim((-15, 10))
    plt.ylim((-10, 15))
    plt.title(title)
    # plt.savefig("img/data.png")
    plt.show()


def generate_one_task(task_ind):
    """
    :param task_ind: task index
    :param plot_data: True or False, default value is False
    :return: saved synthetic task (type=None)

    Code for generating the synthetic data.
    We will have two non-sensitive features and one sensitive feature.
    A sensitive feature value of 0.0 means the example is considered to be in protected group (e.g., female) and 1.0 means it's in non-protected group (e.g., male).
    """
        #32
    # Generate tasks from the 1st distribution
    if task_ind + 1 <= num_tasks / 2:
        mu1, sigma1 = [-7, -7], [[5, 1], [1, 5]]
        mu2, sigma2 = [-3, -3], [[10, 1], [1, 3]]
        X1, y1, z1 = gen_gaussian_task(mu1, sigma1, mu2, sigma2, disc_factor=disc_factor1)
        data_noraml_save(X1, y1, z1, task_ind,False)
    # Generate tasks from the 2nd distribution
    else:
        mu1, sigma1 = [-7, -7], [[5, 1], [1, 5]]
        mu2, sigma2 = [-3, -3], [[10, 1], [1, 3]]
        X2, y2, z2 = gen_gaussian_task(mu1, sigma1, mu2, sigma2, disc_factor=disc_factor2)
        # X2 = X2*(-2)
        data_noraml_save(X2, y2, z2, task_ind,True)

    # visualize the 1st and the 2nd data distribution using 200 data samples
    if task_ind == 0:
        plot_data(X1, y1, z1, 200, "Visualization of the 1st Data Distribution")
    if task_ind + 1 == num_tasks:
        plot_data(X2, y2, z2, 200, "Visualization of the 2nd Data Distribution")


def generate_tasks(save, num_tasks):
    """
    :param save: save directory
    :param num_tasks: number of tasks
    :return: call the function iteratively to generate single synthetic task
    """
    if not os.path.exists(save):
        os.makedirs(save)
    for i in range(num_tasks):
        if i == 0:
            print("================================ 1st Distribution ================================")
        if i == num_tasks / 2:
            print("================================ 2nd Distribution ================================")
        if i % 1 == 0:
            print("generating task: ", i + 1)

        generate_one_task(i)


if __name__ == "__main__":
    a =1
    b =+1 #ax+b
    num_tasks = 64
    disc_factor1 = float(math.pi / 8.0)  # default math.pi / 8.0
    disc_factor2 =float(math.pi / 32.0)  # default math.pi / 8.0
    save = r"./data/data/11-02-2021/normal_64_a="+str(a)+"_b="+str(b)+'_tasks_'+str(num_tasks)+"_disc_factor1_"+str(round(disc_factor1,2))+"_disc_factor2_"+str(round(disc_factor2,2))
    n_samples = 2000  # generate these many data points per task


    generate_tasks(save, num_tasks)
