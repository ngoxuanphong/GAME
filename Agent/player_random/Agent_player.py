import numpy as np

import os
import sys
from main import game_name
from numba import jit, njit, prange
sys.path.append(os.path.abspath(f"base/{game_name}"))
from env import *

# @njit()
def test(p_state, temp_file, per_file):
    arr_action = get_list_action(p_state)
    arr_action = np.where(arr_action == 1)[0]
    act_idx = np.random.randint(0, len(arr_action))
    return arr_action[act_idx], temp_file, per_file

# @njit()
def test2(p_state, temp_file, per_file2):
    arr_action = get_list_action(p_state)
    arr_action = np.where(arr_action == 1)[0]
    act_idx = np.random.randint(0, len(arr_action))
    return arr_action[act_idx], temp_file, per_file2