import numpy as np
import heapq
from scipy.stats.mstats import gmean, hmean
import itertools

import os
import sys
from setup import game_name,time_run_game
sys.path.append(os.path.abspath(f"base/{game_name}"))
from env import *

def agent_hieu_270922(state,file_temp,file_per):
    actions = getValidActions(state)
    actions = np.where(actions == 1)[0]
    action = np.random.choice(actions)
    file_per = (len(state),getActionSize())
    return action,file_temp,file_per

LEN_STATE_hieu_270922,AMOUNT_ACTION_hieu_270922 = normal_main([agent_hieu_270922]*getAgentSize(), 1, [0])[1]

def softmax_hieu_270922(X):
    expo = np.exp(X)
    return expo/np.sum(expo)

def sigmoid_hieu_270922(X):
    return 1/(1+np.exp(-X))

def tanh_hieu_270922(X):
    return np.tanh(X)

def neural_network_hieu_270922(state, file_temp, list_action):
    norm_state = state.copy()
    norm_state = (norm_state/np.linalg.norm(norm_state, 1)).reshape(1,LEN_STATE_hieu_270922)
    norm_state = softmax_hieu_270922(norm_state)
    norm_action = np.zeros(AMOUNT_ACTION_hieu_270922)
    norm_action[list_action] = 1
    norm_action = norm_action.reshape(1, AMOUNT_ACTION_hieu_270922)

    matrixRL1 = norm_state@file_temp[0]
    matrixRL1 = sigmoid_hieu_270922(matrixRL1)          

    matrixRL2 = matrixRL1@file_temp[1]
    matrixRL2 = tanh_hieu_270922(matrixRL2)         

    matrixRL3 = matrixRL2@file_temp[2]
    matrixRL3 = softmax_hieu_270922(matrixRL3)   

    result_val_action = matrixRL3*norm_action
    action_max = np.argmax(result_val_action)
    return action_max


def test(state, file_temp, file_per):
    list_action = getValidActions(state)
    list_action = np.where(list_action == 1)[0]
    if len(file_temp) < 2:
        player = 'Hieu_270922'  #Tên folder của người chơi
        path_save_player = f'system/Agent/{player}/Data/{game_name}_{time_run_game}/'
        with open(f'{path_save_player}best_matrix.npy', 'rb') as outfile:
            best_matrix = np.load(outfile, allow_pickle= True)
        file_temp = best_matrix
    action = neural_network_hieu_270922(state, file_temp, list_action)
    return action, file_temp, file_per

def test2(state, file_temp, file_per, file_per_2):
    list_action = getValidActions(state)
    list_action = np.where(list_action == 1)[0]
    action = neural_network_hieu_270922(state, file_per_2, list_action)
    return action, file_temp, file_per, file_per_2
