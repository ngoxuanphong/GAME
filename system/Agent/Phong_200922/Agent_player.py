import numpy as np
import itertools
import random
import copy
import time
import warnings 
from numba import njit
warnings.filterwarnings('ignore')

import os
import sys
from setup import game_name
sys.path.append(os.path.abspath(f"base/{game_name}"))
from env import *
from system.Data import *
from system.Data2 import *


class Phong_att():
    def __init__(self):
        self.layer = 1
        self.count_matrix = 10
        self.count_test_matrix = 1
        self.add_reward = 1.2
        self.sub_reward = 0.8
        self.percent_matrix_choice = 0.8
        self.count_matrix_remove = 0.6


def player_random(state, temp, per):
    actions = get_list_action(state)
    action = random.choice(actions)
    return action, temp, per

def file_temp_to_action(state, file_temp, layer):
    a = get_list_action(state)
    RELU = np.ones(len(state))
    matrix_new = np.matmul(RELU,file_temp)
    list_val_action = matrix_new[a]
    action = a[np.argmax(list_val_action)]
    return action

def test(state,file_temp,file_per):
    if len(file_temp) < 2:
        file_temp = data_Phong_200922[game_name]
    action = file_temp_to_action(state, file_temp, Phong_att().layer)
    return action,file_temp,file_per