import numpy as np
import os
import sys
from setup import game_name
sys.path.append(os.path.abspath(f"base/{game_name}"))
from env import *
from system.Data import *
from system.Data2 import *
from system.Data3 import *
from system.Data4 import *

if len(sys.argv) == 2:
    game_name = sys.argv[1]
    
INF_AS_FLOAT = np.nan_to_num(np.inf)
LOG_INF = np.log(INF_AS_FLOAT)
HALF_LOG_INF = LOG_INF/2
SQRT_LOG_INF = np.sqrt(LOG_INF)

def Identity(x:np.ndarray):
    return x/np.abs(x).max()

def BinaryStep(x:np.ndarray):
    return np.where(x>=0, 1, 0)

def Sigmoid(x:np.ndarray):
    return 1/(1+np.e**(-np.where(np.abs(x)>LOG_INF, np.sign(x)*LOG_INF, x)))

def SignStep(x:np.ndarray):
    return np.sign(x)

def Tanh(x:np.ndarray):
    x_new = np.where(np.abs(x)>HALF_LOG_INF, np.sign(x)*HALF_LOG_INF, x)
    return (np.e**(2*x_new)-1)/(np.e**(2*x_new)+1)

def ReLU(x:np.ndarray):
    return np.where(x<0, 0, x)/np.max(x)

def SoftPlus(x:np.ndarray):
    x_ = np.where(np.abs(x)>LOG_INF-1, x, np.log(1+np.e**(x)))
    return x_/np.max(x_)

def Gaussian(x:np.ndarray):
    return np.e**(-np.where(np.abs(x)>SQRT_LOG_INF, np.sign(x)*SQRT_LOG_INF, x)**2)

activation_function = [Identity, BinaryStep, Sigmoid, SignStep, Tanh, ReLU, SoftPlus, Gaussian]

def Ann_neural_network(p_state:np.ndarray, data, list_action):
    res_mat = p_state.copy()
    for i in range(len(data)//3):
        res_mat = res_mat @ data[3*i] + data[3*i+1]
        res_mat = np.nan_to_num(res_mat)
        res_mat = activation_function[data[3*i+2]](res_mat)

    res_arr = res_mat[0][list_action]
    a = np.max(res_arr)
    if a >= 0:
        arr_max = np.where(res_arr >= 0.99*a)[0]
    else:
        arr_max = np.where(res_arr >= 1.01*a)[0]

    action_max_idx = np.random.choice(arr_max)
    return list_action[action_max_idx]

def test(p_state, temp_file, per_file):
    if len(temp_file) < 2:
        temp_file = data_An_200922[game_name]
    
    list_action = get_list_action(p_state)
    if temp_file[1] == 0: # fnn
        action = Ann_neural_network(p_state, temp_file[0], list_action)
        return action, temp_file, per_file
    if temp_file[1] == 1: # sg
        if len(temp_file) < 3:
            temp_file.append(temp_file[0][0]/temp_file[0][1])

        res_arr = temp_file[2][list_action]
        a = np.max(res_arr)
        if a >= 0:
            arr_max = np.where(res_arr >= 0.99*a)[0]
        else:
            arr_max = np.where(res_arr >= 1.01*a)[0]

        action_max_idx = np.random.choice(arr_max)
        return list_action[action_max_idx], temp_file, per_file
    if temp_file[1] == 2: # g1
        res_arr = temp_file[0][list_action]
        a = np.max(res_arr)
        if a >= 0:
            arr_max = np.where(res_arr >= 0.99*a)[0]
        else:
            arr_max = np.where(res_arr >= 1.01*a)[0]

        action_max_idx = np.random.choice(arr_max)
        return list_action[action_max_idx], temp_file, per_file
    if temp_file[1] == 3: # bnn
        res_ = p_state @ temp_file[0]
        res_arr = res_[list_action]
        a = np.max(res_arr)
        if a >= 0:
            arr_max = np.where(res_arr >= 0.99*a)[0]
        else:
            arr_max = np.where(res_arr >= 1.01*a)[0]

        action_max_idx = np.random.choice(arr_max)
        return list_action[action_max_idx], temp_file, per_file