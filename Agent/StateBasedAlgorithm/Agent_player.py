# state based
# small NN deep
import numpy as np
import random as rd
from numba import njit, jit
import sys, os
from setup import SHOT_PATH
import importlib.util
game_name = sys.argv[1]

def setup_game(game_name):
    spec = importlib.util.spec_from_file_location('env', f"{SHOT_PATH}base/{game_name}/env.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module 
    spec.loader.exec_module(module)
    return module

env = setup_game(game_name)

getActionSize = env.getActionSize
getStateSize = env.getStateSize
getAgentSize = env.getAgentSize

getValidActions = env.getValidActions
getReward = env.getReward

normal_main = env.normal_main
numba_main_2 = env.numba_main_2

from numba.typed import List

def DataAgent():
    per = [np.zeros((100,getStateSize(),getActionSize())), #[0][value][idS] khi mở đầu game
           np.zeros((100,getStateSize(),getActionSize())), #[1][value][ids] lưu lại cuối cùng
           np.zeros((1,1,1)) #[2][0][0][0] vừa thắng hay thua
           ]
    return per

@njit()
def Train(state,per):
    actions = getValidActions(state)
    weights = np.zeros(getActionSize())
    if per[2][0][0][0] == 0:
        temp = np.argsort(np.argsort(np.random.rand(getActionSize()))) * 1.0
        weights += temp
        for ids in range(getStateSize()):
            if state[ids] < 100:
                per[0][int(state[ids])][ids] += temp/np.max(temp)
    else:
        for ids in range(getStateSize()):
            if state[ids] < 100:
                weights += np.argsort(np.argsort(per[0][int(state[ids])][ids]))/getActionSize()
    output = weights * actions + actions
    action = np.argmax(output)
    win = getReward(state)
    if win != -1:
        if win == 1:
            # print("đây")
            per[1] += per[0]
            per[2][0][0][0] = 1
        else:
            per[0] = np.zeros((100,getStateSize(),getActionSize()))
            per[2][0][0][0] = 0
    return action, per

@njit()
def Test(state,per):
    actions = getValidActions(state)
    weights = np.zeros(getActionSize())
    for ids in range(getStateSize()):
        if state[ids] < 100:
            weights += np.argsort(np.argsort(per[1][int(state[ids])][ids]))/getActionSize()
    output = weights * actions + actions
    action = np.argmax(output)
    return action,per