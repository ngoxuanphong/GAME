import random as rd
import numpy as np
from numba import njit
from numba.typed import List
import os
import sys
from setup import game_name,time_run_game
sys.path.append(os.path.abspath(f"base/{game_name}"))
from env import *

@njit()
def test2(state,temp,per):
    arr_action = get_list_action(state)
    mt = np.dot(state,per[0])
    mt *= mt>0
    mt = np.dot(mt,per[1])
    mt *= arr_action
    if check_victory(state) == 1:
        per[2][0][0] += 1
    if check_victory(state) == 0:
        per[2][0][1] += 1
    if per[2][0][0] > 0 and per[2][0][1] > 100 and per[2][0][0]/per[2][0][1] < 0.2:
        per = List()
        per.append((np.random.random((len(state),100)))*2 -0.6)
        per.append(np.random.random((100,amount_action())))
        per.append(np.zeros((1,2)))
    if np.argmax(mt) not in np.where(arr_action == 1)[0]:
        action = np.where(arr_action == 1)[0][0]
    else: action = np.argmax(mt)
    return action, temp,per