import numpy as np
from env import *
import random
import os
import sys
from setup import game_name,time_run_game
sys.path.append(os.path.abspath(f"base/{game_name}"))

def test(state,temp,per):
    if len(temp)<2:
        player = 'Dat_130922'  #Tên folder của người chơi
        path_save_player = f'system/Agent/{player}/Data/{game_name}_{time_run_game}/'
        data = np.load(os.path.join(path_save_player,'Best.npz'))
        temp = [data['w1'],data['w2']]

    list_action = getValidActions(state)
    list_action = np.where(list_action == 1)[0]
    hidden1 = state.dot(temp[0])
    hidden2 = hidden1 * (hidden1>0)
    values = hidden2.dot(temp[1])
    action = list_action[np.argmax(values[list_action])]
    return action,temp,per

def test2(state,temp,per, file_per_2):
    list_action = getValidActions(state)
    list_action = np.where(list_action == 1)[0]
    hidden1 = state.dot(file_per_2[0])
    hidden2 = hidden1 * (hidden1>0)
    values = hidden2.dot(file_per_2[1])
    action = list_action[np.argmax(values[list_action])]
    return action,temp,per, file_per_2