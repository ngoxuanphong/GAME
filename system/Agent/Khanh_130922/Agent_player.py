import random
import numpy as np

random.seed(10)
import os
import sys
from setup import game_name,time_run_game
sys.path.append(os.path.abspath(f"base/{game_name}"))
from env import *

from numba import jit, njit, prange
if len(sys.argv) == 2:
    game_name = sys.argv[1]

@jit()
def matmul(A, B):
    if A.shape == (len(A),):
        A = np.array([A])
    m = A.shape[0]
    n = A.shape[1]
    p = B.shape[1]
    C = np.zeros((m,p))
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i,j] = C[i,j] + A[i,k]*B[k,j]
    return C

@njit()
def _sigmoid_khanh_130922_(x):
    sig = 1 / (1 + np.exp(-x))
    return sig

@njit()
def _silu_khanh_130922_(x, theda = 1.0):
    return x * _sigmoid_khanh_130922_(theda *x)

# @jit()
def neural_network_khanh_130922(play_state, file_temp):
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
        matran1 = _silu_khanh_130922_(matran1, theda = 1.0)
        matran2 = np.matmul(matran1, file_temp[1])
        return matran2
    else :#SUSHIGO-main, MACHIKOR0
        matran1 = np.matmul(play_state, file_temp[0])
        matran1 *= (matran1 > 0)
        matran2 = np.matmul(matran1, file_temp[1])
        return matran2


def test(play_state,file_temp,file_per):
    a = getValidActions(play_state)
    a = np.where(a == 1)[0]
    if len(file_temp) < 2:
        player = 'Khanh_130922'  #Tên folder của người chơi
        path_save_player = f'system/Agent/{player}/Data/{game_name}_{time_run_game}/'
        file_temp = np.load(f'{path_save_player}CK_Win.npy',allow_pickle=True)
    matran2 = neural_network_khanh_130922(play_state, file_temp)
    max_ = 0
    action_max = a[random.randrange(len(a))]
    
    for act in a:
        if matran2[act] > max_:
            max_ = matran2[act]
            action_max = act
    return action_max,file_temp,file_per

def test2(play_state,file_temp,file_per, file_per_2):
    a = getValidActions(play_state)
    a = np.where(a == 1)[0]
    matran2 = neural_network_khanh_130922(play_state, file_per_2)
    max_ = 0
    action_max = a[random.randrange(len(a))]
    
    for act in a:
        if matran2[act] > max_:
            max_ = matran2[act]
            action_max = act
    return action_max,file_temp,file_per, file_per_2