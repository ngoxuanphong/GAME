import numpy as np

import os
import sys
from setup import game_name
sys.path.append(os.path.abspath(f"base/{game_name}"))
from env import *

player = 'Phong_08_12_2022'
path_data = f'Agent/{player}/Data'
if not os.path.exists(path_data):
    os.mkdir(path_data)
path_save_player = f'Agent/{player}/Data/{game_name}/'
if not os.path.exists(path_save_player):
    os.mkdir(path_save_player)

def test_(p_state, temp_file, per_file):
    arr_action = get_list_action(p_state)
    act_idx = np.random.randint(0, len(arr_action))
    return arr_action[act_idx], temp_file, per_file

def train_(state, temp_file, per_file):
    arr_action = get_list_action(state)
    act_idx = np.random.randint(0, len(arr_action))
    if check_victory(state) != -1:
        per_file[0] += 1
    return arr_action[act_idx], temp_file, per_file

def train(n):
    list_player = [test_]*(amount_player() - 1) + [train_]
    c, p = normal_main(list_player, n*1000, [0])
    print(c, p)
    np.save(f'{path_save_player}per.npy', p)
    np.save(f'{path_save_player}count.npy', np.array(c))