import numpy as np
from numba import njit
import os
import sys
from setup import game_name,time_run_game
sys.path.append(os.path.abspath(f"base/{game_name}"))
from env import *

@njit()
def basic_act_NhatAnh_270922(state,base):
    actions = get_list_action(state)
    actions = np.where(actions == 1)[0]
    for act in base:
        if act in actions:
            return act
    ind = np.random.randint(len(actions))
    action = actions[ind]
    return action


def advance_act_NhatAnh_270922(state,data):
    for id in range(len(data[1])):
        mt = np.dot(state,data[1][id])
        if mt[0] <= 0:
            action = basic_act_NhatAnh_270922(state,data[0][id-1])
            return action
        else:
            action = basic_act_NhatAnh_270922(state,data[0][id])
    return action

def test(state,temp,per):
    if len(temp) < 2:
        player = 'NhatAnh_270922'
        path_save_player = f'system/Agent/{player}/Data/{game_name}_{time_run_game}/'
        temp = np.load(f'{path_save_player}best.npy',allow_pickle=True)
    action = advance_act_NhatAnh_270922(state,temp)
    return action,temp,per

def test2(state,temp,per, file_per_2):
    action = advance_act_NhatAnh_270922(state,file_per_2)
    return action,temp,per, file_per_2