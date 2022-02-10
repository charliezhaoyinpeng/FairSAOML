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
    parser.add_argument("--K", default=100, type=int, help="few shots in support") #100 500 600 700 FFML batch 90 iter 25 AOD
    parser.add_argument("--Kq", default=200, type=int, help="few shots in query")
    parser.add_argument("--inner_steps", default=1, type=int, help="gradient steps in the inner loop")
    parser.add_argument("--pd_updates", default=1, type=int, help="inner primal-dual iteration")
    parser.add_argument("--eta_1", default=5, type=float, help="step size of inner primal update for validation model")
    parser.add_argument("--eta_2", default=0.000050, type=float, help="step size of inner dual update for validation model")
    parser.add_argument("--meta_eta_1", default=5, type=float,
                        help="step size for outer primal update")
    parser.add_argument("--meta_eta_2", default=0.000050, type=float,
                        help="step size for outer dual update")
    parser.add_argument("--num_iterations", default=1, type=int, help="outer iteration")
    # parser.add_argument("--xi", default=0.015, type=float, help="parameter of outer regularization")
    parser.add_argument("--delta", default=10, type=int, help="some constant for outer augmentation")
    parser.add_argument("--eps", default=0.05, type=float, help="fairness threshold")
    parser.add_argument("--d_feature", default=3, type=int, help="feature size of the data set.")
    parser.add_argument("--data_path", default=r'.\data\data', type=str, help="root to all dataset")
    parser.add_argument("--dataset", default=r'movie_length_90task', type=str, help="dataset folder name")
    parser.add_argument("--save", default=r'movie_length_90task', type=str, help="save location")
    # parser.add_argument("--meta_batch", default=5,type=int, help="save location")
    parser.add_argument("--radius", default=10, type=int, help="radius")
    parser.add_argument("--net_dim", default=7, type=int, help="net_dim")

    parser = parser.parse_args()
    return  parser


if __name__ == "__main__":
    lamb = random.choice([0.0001, 0.001, 0.01, 0.1, 1])  # lambda initialization
    parser = get_parameter()


    val_batch_size = parser.val_batch_size  # data points for validation
    K = parser.K  # few shots in support
    Kq = parser.Kq  # shots for query
    inner_steps = parser.inner_steps  # gradient steps in the inner loop
    pd_updates = parser.pd_updates  # inner primal-dual iteration
    num_iterations = parser.num_iterations  # outer iteration
    # meta_batch = parser.meta_batch
    delta = parser.delta

    eta_1 = parser.eta_1  # step size of inner primal update
    eta_2 = parser.eta_2  # step size of inner dual update
    meta_eta_1= parser.meta_eta_1
    meta_eta_2 = parser.meta_eta_2


    eps = parser.eps  # fairness threshold
    # eps = 0.35  # new fairness threshold
    num_neighbors = 3


    d_feature = parser.d_feature  # feature size of the data set
    data_path = parser.data_path
    dataset = parser.dataset
    radius = parser.radius
    net_dim = parser.net_dim
    save = parser.save
    tasks = [x[0] for x in os.walk(os.path.join(data_path, dataset))][1:]



    dp,eop,acc,loss,dbc,loss_plus_dbc=fairSAOML(d_feature, lamb, tasks, data_path, dataset, save,
            K, Kq, val_batch_size, num_neighbors,
            num_iterations, inner_steps, pd_updates,
            eta_1, eta_2, eps,radius,meta_eta_1,meta_eta_2,delta,net_dim)





