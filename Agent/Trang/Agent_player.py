import random as rd
import numpy as np

import os
import sys
from setup import game_name,time_run_game
sys.path.append(os.path.abspath(f"base/{game_name}"))
from env import *

def test2(play_state,file_temp,file_per):
    arr_action = get_list_action(play_state)
    arr_action = np.where(arr_action == 1)[0]
    act_idx = np.random.randint(0, len(arr_action))
    
    if type(file_per) == int:
        file_per = [[(np.random.random((len(play_state),100)))*2 -0.6,np.random.random((100,amount_action()))], 0, 0]
    if check_victory(play_state) == 1:
        file_per[1] +=1
    if check_victory(play_state) == -1:  
        file_per[2] +=1
    if file_per[2]>0:
        if file_per[1]/file_per[2] <0.2:
            file_per = [[(np.random.random((len(play_state),100)))*2 -0.6,np.random.random((100,amount_action()))], 0, 0]
    return arr_action[act_idx], file_temp,file_per