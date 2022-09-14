import numpy as np

import os
import sys
from main import game_name
sys.path.append(os.path.abspath(f"base/{game_name}"))
from env import *

def test(p_state, temp_file, per_file):
    arr_action = get_list_action(p_state)
    act_idx = np.random.randint(0, len(arr_action))
    return arr_action[act_idx], temp_file, per_file