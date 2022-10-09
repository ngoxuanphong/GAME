import numpy as np
import os
import sys

from setup import game_name,time_run_game
sys.path.append(os.path.abspath(f"base/{game_name}"))
from env import *

def Identity_an_130922(x):
    return x

def BinaryStep_an_130922(x):
    x[x>=0] = 1.0
    x[x<0] = 0.0
    return x

def Sigmoid_an_130922(x):
    return 1.0 / (1.0 + np.e**(-x))

def NegativePositiveStep_an_130922(x):
    x[x>=0] = 1.0
    x[x<0] = -1.0
    return x

def Tanh_an_130922(x):
    return (np.e**(x) - np.e**(-x)) / (np.e**(x) + np.e**(-x))

def ReLU_an_130922(x):
    return x * (x>0)

def LeakyReLU_an_130922(x):
    x[x<0] *= 0.01
    return x

def PReLU_an_130922(x, a=0.5):
    x[x<0] *= 0.5
    return x

def Gaussian_an_130922(x):
    return np.e**(-x**2)

list_activation_function = [Identity_an_130922, BinaryStep_an_130922, Sigmoid_an_130922, NegativePositiveStep_an_130922, Tanh_an_130922, ReLU_an_130922, LeakyReLU_an_130922, PReLU_an_130922, Gaussian_an_130922]

def neural_network_an_130922(p_state, data, list_action):
    res_mat = p_state.copy()
    for i in range(len(data)):
        if i % 2 == 0:
            res_mat = res_mat @ data[i]
            max_x = np.max(np.abs(res_mat))
            max_x_1 = max_x/25
            res_mat = res_mat / max_x_1
        else:
            res_mat = list_activation_function[data[i]](res_mat)
    
    res_arr = res_mat[list_action]
    arr_max = np.where(res_arr == np.max(res_arr))[0]
    action_max_idx = np.random.choice(arr_max)
    return list_action[action_max_idx]

def test(p_state, temp_file, per_file):
    if len(temp_file) < 2:
        player = 'An_130922'
        path_save_player = f'system/Agent/{player}/Data/{game_name}_{time_run_game}/'
        temp_file = np.load(f'{path_save_player}/Ahih1st.npy', allow_pickle=True)
    
    list_action = get_list_action(p_state)
    list_action = np.where(list_action == 1)[0]
    action = neural_network_an_130922(p_state, temp_file, list_action)
    return action, temp_file, per_file

def test2(p_state, temp_file, per_file, file_per_2):
    list_action = get_list_action(p_state)
    list_action = np.where(list_action == 1)[0]
    action = neural_network_an_130922(p_state, file_per_2, list_action)
    return action, temp_file, per_file, file_per_2
