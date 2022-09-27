import math
import numpy as np
import heapq
from scipy.stats.mstats import gmean, hmean
import json
import random

random.seed(100)
import os
import sys
from setup import game_name
sys.path.append(os.path.abspath(f"base/{game_name}"))
from env import *
from system.Data import *
from system.Data2 import *
from system.Data3 import *
from system.Data4 import *

def silu(x, theda = 1.0):
    return x * sigmoid(theda *x)
def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))
    return sig


def test(state, file_temp, file_per):
    list_action = get_list_action(state)
    if len(file_temp) < 2:
        file_temp = data_Hieu_130922[game_name]
    action = neural_network(state, file_temp, list_action)
    return action, file_temp, file_per


def neural_network(state, file_temp, list_action):
    norm_state = (state/np.linalg.norm(state, 1)).reshape(1,len(state))
    # print('check2', norm_state)
    # norm_state = 1 / (1 + np.exp(-norm_state))          #dạng sigmoid
    # norm_state = np.arctan(norm_state)                     #dạng sin
    norm_state = np.tanh(norm_state)                    #dạng tanh
    # print('check3', norm_state)
    norm_action = np.zeros(amount_action())
    norm_action[list_action] = 1
    norm_action = norm_action.reshape(1, amount_action())
    # print(matrix1.shape)
    matrixRL1 = norm_state@file_temp[0]
    # print(matrixRL1.shape)
    # matrixRL1 = silu(matrixRL1, theda = 1.0)
    matrixRL1 = matrixRL1*(matrixRL1 > 0)           #activation = relu
    # print(matrixRL1.shape, file_temp[1].shape)
    matrixRL2 = matrixRL1@file_temp[1]
    matrixRL2 = 1 / (1 + np.exp(-matrixRL2))            #activation = sigmoid
    # matrixRL2 = matrixRL2*(matrixRL2 > 0)           #activation = relu
    matrixRL3 = matrixRL2@file_temp[2]
    matrixRL3 = np.tanh(matrixRL3)              #activation = tanh
    # matrixRL3 = np.sin(matrixRL3)               #activation = sin
    result_val_action = matrixRL3*norm_action
    # print(matrixRL3.shape)
    action_max = np.argmax(result_val_action)
    # print(matrixRL3)
    return action_max
