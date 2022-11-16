import random as rd
import numpy as np
import os
import sys
from setup import game_name,time_run_game
sys.path.append(os.path.abspath(f"base/{game_name}"))
from env import *

def data_to_layer_NhatAnh130922(state,data):
    for ind in data[0]:
        state = np.dot(state,ind)
        state *= state > 0
    active = state > 0
    layer = data[1] * active
    return layer
def test(state,temp,per):
    if len(temp) < 2:
        player = 'NhatAnh_130922'
        path_save_player = f'system/Agent/{player}/Data/{game_name}_{time_run_game}/'
        best = np.load(f'{path_save_player}best.npy',allow_pickle=True)
        temp = [list(best),0]
    layer = np.zeros(getActionSize())
    for data in temp[0]:
        layer += data_to_layer_NhatAnh130922(state,data)
    base = np.zeros(getActionSize())
    actions = getValidActions(state)
    actions = np.where(actions == 1)[0]
    for act in actions:
        base[act] = 1
    layer *= base
    base += layer
    action = np.random.choice(np.where(base == np.max(base))[0])
    return action,temp,per

def test2(state,temp,per, file_per_2):
    layer = np.zeros(getActionSize())
    for data in file_per_2[0]:
        layer += data_to_layer_NhatAnh130922(state,data)
    base = np.zeros(getActionSize())
    actions = getValidActions(state)
    actions = np.where(actions == 1)[0]
    for act in actions:
        base[act] = 1
    layer *= base
    base += layer
    action = np.random.choice(np.where(base == np.max(base))[0])
    return action,temp,per, file_per_2