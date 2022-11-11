import math
import numpy as np
import heapq
from scipy.stats.mstats import gmean, hmean
import json
import random

random.seed(100)
import os
import sys
from setup import game_name,time_run_game
sys.path.append(os.path.abspath(f"base/{game_name}"))
from env import *


def neural_network_hieu_130922(state, file_temp, list_action):
    norm_state = (state/np.linalg.norm(state, 1)).reshape(1,len(state))
    norm_state = np.tanh(norm_state)                    #dạng tanh
    norm_action = np.zeros(getActionSize())
    norm_action[list_action] = 1
    norm_action = norm_action.reshape(1, getActionSize())
    matrixRL1 = norm_state@file_temp[0]
    matrixRL1 = matrixRL1*(matrixRL1 > 0)           #activation = relu
    matrixRL2 = matrixRL1@file_temp[1]
    matrixRL2 = 1 / (1 + np.exp(-matrixRL2))            #activation = sigmoid
    matrixRL3 = matrixRL2@file_temp[2]
    matrixRL3 = np.tanh(matrixRL3)              #activation = tanh
    result_val_action = matrixRL3*norm_action
    action_max = np.argmax(result_val_action)
    return action_max

def test(state, file_temp, file_per):
    list_action = getValidActions(state)
    list_action = np.where(list_action == 1)[0]
    if len(file_temp) < 2:
        player = 'Hieu_130922'  #Tên folder của người chơi
        path_save_player = f'system/Agent/{player}/Data/{game_name}_{time_run_game}/'
        with open(f'{path_save_player}CK_Matran.npy', 'rb') as outfile:
            best_matrix = np.load(outfile, allow_pickle= True)
        file_temp = best_matrix
    action = neural_network_hieu_130922(state, file_temp, list_action)
    return action, file_temp, file_per

def test2(state, file_temp, file_per, file_per_2):
    list_action = getValidActions(state)
    list_action = np.where(list_action == 1)[0]
    action = neural_network_hieu_130922(state, file_per_2, list_action)
    return action, file_temp, file_per, file_per_2