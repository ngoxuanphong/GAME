import numpy as np
from env import *
import random
import json
import os
import sys
from main import game_name
from system.Data import *
sys.path.append(os.path.abspath(f"base/{game_name}"))
from system.Data import *
from system.Data2 import *
from system.Data3 import *
from system.Data4 import *
if len(sys.argv) == 2:
    game_name = sys.argv[1]
    
def test(state,temp,per):
    if len(temp)<2:
        temp = data_Dat_200922[game_name]
    list_action = get_list_action(state)
    hidden1 = state.dot(temp[0])
    hidden2 = hidden1 * (hidden1>0)
    values = hidden2.dot(temp[1])
    action = list_action[np.argmax(values[list_action])]
    return action,temp,per
    
    


