import torch
import math
import numpy as np
from random import random as rd


def cal_discrimination(input_zy):
    a_values = []
    b_values = []
    for line in input_zy:
        if line[0] == 0:
            a_values.append(line[1])
        elif line[0] == 1:
            b_values.append(line[1])

    if len(a_values) == 0:
        discrimination = sum(b_values) * 1.0 / len(b_values)
    elif len(b_values) == 0:
        discrimination = sum(a_values) * 1.0 / len(a_values)
    else:
        discrimination = sum(a_values) * 1.0 / len(a_values) - sum(b_values) * 1.0 / len(b_values)
    return abs(discrimination)


# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return math.sqrt(distance)


# Locate the most similar neighbors
# example: get_neighbors(yX, X[0], 3)
def get_neighbors(yX, target_row, num_neighbors):
    distances = list()
    for yX_row in yX:
        X_row = yX_row[1:]
        y = yX_row[0]
        dist = euclidean_distance(target_row, X_row)
        distances.append((y, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


def cal_dbc(input_zy):
    length = len(input_zy)
    z_bar = np.mean(input_zy[:, 0])
    dbc = 0
    for zy in input_zy:
        dbc += (zy[0] - z_bar) * zy[1] * 1.0
    return abs(dbc / length)

# def cal_dbc(input_zy):
#     length = len(input_zy)
#     count1=0
#     for item in input_zy:
#         if item[0]==0:
#             count1 +=1
#     p1 = (count1*1.0)/length;
#     sum =0;
#     for item in input_zy:
#         z = item[0];
#         y_hat = item[1];
#         sum += ((z+1)/2-p1)*y_hat;
#     sum = sum*p1(1-p1);
#
#     return sum / length

def cal_dbc(z,yhat):
    print("z",z)
    print("yhat",yhat)
    if len(z)!=len(yhat):
        print("dbc error")
        return
    length = len(z)
    countz =0;
    for item in z:
        if item==1:
            # print("noe1",item)
            countz +=1;
    p1 = (countz*1.0)/length;
    print("count_z",countz)
    print("p1", p1)
    print("length",length)
    sum = 0;
    for i in range(len(z)):
        cur_z = z[i]
        cur_y_hat = yhat[i]
        # print(cur_z,y_hat)
        sum += ((cur_z + 1) / 2 - p1) * cur_y_hat
    sum = sum * 1/(p1*(1 - p1));
    output = sum / length
    output = torch.abs(output)

    return output

def cal_dp(input_zy, t, xi):
    count1 = 0
    count2 = 0
    for item in input_zy:
        if item[0] == 0:
            count1 += 1
            if item[1].round() == 1:
                count2 += 1
    try:
        dp = abs(1 - count2 * 1.0 / count1) + t * rd() * xi
    except:
        dp = 0
    return dp


def cal_eop(z_y_hat_y, t, xi):
    count1 = 0
    count2 = 0
    for item in z_y_hat_y:
        if item[0] == 0 and item[2] == 1:
            count1 += 1
            if item[1].round() == 1:
                count2 += 1
    try:
        eop = abs(1 - count2 * 1.0 / count1) + t * rd() * xi
    except:
        eop = 0
    return eop


def cal_consistency(yX, num_neighbors):
    ans = 0
    for yX_row in yX:
        temp = 0
        target_row = yX_row[1:]
        target_y = yX_row[0]
        y_neighbors = get_neighbors(yX, target_row, num_neighbors)
        for y_neighbor in y_neighbors:
            temp += abs(target_y - y_neighbor)
        ans += temp
    return (1 - (ans * 1.0) / (len(yX) * num_neighbors))
