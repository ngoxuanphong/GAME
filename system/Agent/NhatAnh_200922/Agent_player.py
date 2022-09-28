import random as rd
import numpy as np
from numba import njit
from numba import cuda
# import tensorflow as tf

import os
import sys
from setup import game_name
sys.path.append(os.path.abspath(f"base/{game_name}"))
from env import *
from system.Data import *
from system.Data2 import *
from system.Data3 import *
from system.Data4 import *

if len(sys.argv) == 2:
    game_name = sys.argv[1]
    
@njit()
def basic_act(state,base):
    actions = get_list_action(state)
    for act in base:
        if act in actions:
            return act
    ind = np.random.randint(len(actions))
    action = actions[ind]
    return action

def test(state,temp,per):
    if len(temp) < 2:
        temp = data_NhatAnh_200922[game_name]
    action = basic_act(state,temp)
    return action,temp,per
