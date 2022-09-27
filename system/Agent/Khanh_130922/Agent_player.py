import random
import numpy as np

random.seed(10)
import os
import sys
from setup import game_name,time_run_game
sys.path.append(os.path.abspath(f"base/{game_name}"))
from env import *
from system.Data import *
from system.Data2 import *
from system.Data3 import *
from system.Data4 import *
def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))
    return sig

def silu(x, theda = 1.0):
    return x * sigmoid(theda *x)


def neural_network(play_state, file_temp):
    if 55 < len(play_state) < 70 or len(play_state) > 250 : # TLMN , TLMN_v2 , CENTURY
        matran1 = np.matmul(play_state,file_temp[0])
        matran1 = 1 / (1 + np.exp(-matran1))
        matran21 = np.matmul(matran1,file_temp[1])
        matran21 *= (matran21 > 0)
        matran2 = np.matmul(matran21, file_temp[2])
        return matran2    
    elif 120 <len(play_state)  < 170:# SPLENDOr SPlendor_view_only
        matrix1 = play_state@file_temp[0]
        matrixRL1 = 1 / (1 + np.exp(-matrix1))
        matrix2 = matrixRL1@file_temp[1]
        matrixRL2 = 1 / (1 + np.exp(-matrix2))
        all_action_val = matrixRL2@file_temp[2]
        return all_action_val  
    elif 170 < len(play_state) < 250  : #SHERIFF 
        matran1 = np.matmul(play_state, file_temp[0])
        matran1 = silu(matran1, theda = 1.0)
        matran2 = np.matmul(matran1, file_temp[1])
        return matran2
    else :#SUSHIGO-main, MACHIKOR0
        matran1 = np.matmul(play_state, file_temp[0])
        matran1 *= (matran1 > 0)
        matran2 = np.matmul(matran1, file_temp[1])
        return matran2

def test(play_state,file_temp,file_per):
    a = get_list_action(play_state)
    
    if len(file_temp) < 2:
        file_temp = data_Khanh_130922[game_name]

    matran2 = neural_network(play_state, file_temp)
    max_ = 0
    action_max = a[random.randrange(len(a))]
    
    for act in a:
        if matran2[act] > max_:
            max_ = matran2[act]
            action_max = act
    return action_max,file_temp,file_per
