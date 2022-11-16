import numpy as np
import os
import sys
from setup import game_name,time_run_game
sys.path.append(os.path.abspath(f"base/{game_name}"))
from env import *

def Identity_an_270922(x:np.ndarray):
    return x/np.abs(x).max()

def BinaryStep_an_270922(x:np.ndarray):
    return np.where(x>=0, 1, 0)

def Sigmoid_an_270922(x:np.ndarray):
    INF_AS_FLOAT = np.nan_to_num(np.inf)
    LOG_INF = np.log(INF_AS_FLOAT)
    return 1/(1+np.e**(-np.where(np.abs(x)>LOG_INF, np.sign(x)*LOG_INF, x)))

def SignStep_an_270922(x:np.ndarray):
    return np.sign(x)

def Tanh_an_270922(x:np.ndarray):
    INF_AS_FLOAT = np.nan_to_num(np.inf)
    LOG_INF = np.log(INF_AS_FLOAT)
    HALF_LOG_INF = LOG_INF/2
    x_new = np.where(np.abs(x)>HALF_LOG_INF, np.sign(x)*HALF_LOG_INF, x)
    return (np.e**(2*x_new)-1)/(np.e**(2*x_new)+1)

def ReLU_an_270922(x:np.ndarray):
    return np.where(x<0, 0, x)/np.max(x)

def LeakyReLU_an_270922(x:np.ndarray):
    x_new = np.where(x<0, 0.01*x, x)
    return x_new/np.abs(x_new).max()

def PReLU_an_270922(x:np.ndarray):
    x_new = np.where(x<0, 0.5*x, x)
    return x_new/np.abs(x_new).max()

def SoftPlus_an_270922(x:np.ndarray):
    x_new = np.where(np.abs(x)>LOG_INF-1e-9, x, np.log(1+np.e**(x)))
    return x_new/np.max(x_new)

def Gaussian_an_270922(x:np.ndarray):
    INF_AS_FLOAT = np.nan_to_num(np.inf)
    LOG_INF = np.log(INF_AS_FLOAT)
    HALF_LOG_INF = LOG_INF/2
    SQRT_LOG_INF = np.sqrt(LOG_INF)
    return np.e**(-np.where(np.abs(x)>SQRT_LOG_INF, np.sign(x)*SQRT_LOG_INF, x)**2)

activation_function = [Identity_an_270922, BinaryStep_an_270922, Sigmoid_an_270922, SignStep_an_270922, Tanh_an_270922, ReLU_an_270922, LeakyReLU_an_270922, PReLU_an_270922, SoftPlus_an_270922, Gaussian_an_270922]

def Ann_neural_network_an_270922(p_state:np.ndarray, data, list_action):
    res_mat = p_state.copy()
    for i in range(len(data)//3):
        res_mat = res_mat @ data[3*i] + data[3*i+1]
        res_mat = np.nan_to_num(res_mat)
        res_mat = activation_function[data[3*i+2]](res_mat)
    
    res_arr = res_mat[list_action]
    a = np.max(res_arr)
    if a >= 0:
        arr_max = np.where(res_arr >= 0.99*a)[0]
    else:
        arr_max = np.where(res_arr >= 1.01*a)[0]
    
    return list_action[np.random.choice(arr_max)]

def test(p_state, temp_file, per_file):
    if len(temp_file) < 2:
        player = 'An_270922'
        path_save_player = f'system/Agent/{player}/Data/{game_name}_{time_run_game}/'   
        temp_file = np.load(f'{path_save_player}Ahih1st.npy', allow_pickle=True)
    
    list_action = getValidActions(p_state)
    list_action = np.where(list_action == 1)[0]
    if temp_file[1] == 0:
        action = Ann_neural_network_an_270922(p_state, temp_file[0], list_action)
        return action, temp_file, per_file
    if temp_file[1] == 1:
        if len(temp_file) < 3:
            temp_file.append(temp_file[0][0]/temp_file[0][1])
        
        res_arr = temp_file[2][list_action]
        a = np.max(res_arr)
        arr_max = np.where(res_arr >= 0.99*a)[0]
        
        return list_action[np.random.choice(arr_max)], temp_file, per_file

def test2(p_state, temp_file, per_file, file_per_2):

    list_action = getValidActions(p_state)
    list_action = np.where(list_action == 1)[0]
    if file_per_2[1] == 0:
        action = Ann_neural_network_an_270922(p_state, file_per_2[0], list_action)
        return action, file_per_2, per_file, file_per_2
    if file_per_2[1] == 1:
        if len(file_per_2) < 3:
            file_per_2.append(file_per_2[0][0]/file_per_2[0][1])
        
        res_arr = file_per_2[2][list_action]
        a = np.max(res_arr)
        arr_max = np.where(res_arr >= 0.99*a)[0]
        
        return list_action[np.random.choice(arr_max)], temp_file, per_file, file_per_2
