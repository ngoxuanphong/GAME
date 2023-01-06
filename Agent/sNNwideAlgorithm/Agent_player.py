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
    per = [np.random.randint(0,20,size = (getStateSize(),getStateSize()//2)) * 1/10 - 1, # [0] layer 1
        np.random.randint(0,20,size = (getStateSize()//2,getActionSize())) * 1/10 - 1, # [1] layer 2
        np.random.randint(0,20,size = (getActionSize(),getStateSize())) * 1/10 - 1, # [2] layer 3
        np.random.randint(0,20,size = (getStateSize(),getActionSize())) * 1/10 - 1, # [3] layer 4
        np.zeros((1,3)), # [4][0][0]: số trận đã chơi, [4][0][1]: số trận thắng, [4][0][2]: tỉ lệ thắng best
        np.zeros((getStateSize(),getStateSize()//2)), # [5] best layer 1
        np.zeros((getStateSize()//2,getActionSize())), # [6] best layer 2
        np.zeros((getActionSize(),getStateSize())), # [7] best layer 3
        np.zeros((getStateSize(),getActionSize())) # [8] best layer 4
            ]
    return per

@njit()
def Train(state,per):
    timesPlayed = per[4][0]
    actions = getValidActions(state)
    # tempState = np.zeros((1,getStateSize())) + state
    output = state @ per[0]
    output *= (output > 0) *1.0
    output = output @ per[1]
    output *= (output > 0) *1.0
    output = output @ per[2]
    output = (output > 0) *1.0
    output = output @ per[3]
    output = (output > 0) *1.0
    output = output * actions + actions
    action = np.argmax(output)
    win = getReward(state)
    if win != -1:
        # hết game count + 1
        per[4][0][0] += 1
        if win == 1:
            # thắng count + 1
            per[4][0][1] += 1
            # chơi đủ 10k trận, xét độ mạnh
        if per[4][0][0] == 10000:
            if per[4][0][1] > per[4][0][2]:
                per[4][0][2] = per[4][0][1]
                per[5] = per[0]
                per[6] = per[1]
                per[7] = per[2]
                per[8] = per[3]
            # reset bộ đếm, tạo ma trận mới
            per[4][0][0] = 0
            per[4][0][1] = 0
            per[0] = np.random.randint(0,20,size = (getStateSize(),getStateSize()//2)) * 1/10 - 1 # [0] layer 1
            per[1] = np.random.randint(0,20,size = (getStateSize()//2,getActionSize())) * 1/10 - 1 # [1] layer 2
            per[2] = np.random.randint(0,20,size = (getActionSize(),getStateSize())) * 1/10 - 1 # [2] layer 3
            per[3] = np.random.randint(0,20,size = (getStateSize(),getActionSize())) * 1/10 - 1 # [3] layer 4
    return action, per

@njit()
def Test(state,per):
    actions = getValidActions(state)
    tempState = np.zeros((1,getStateSize())) + state
    output = tempState @ per[5]
    output *= (output > 0) *1.0
    output = output @ per[6]
    output *= (output > 0) *1.0
    output = output @ per[7]
    output *= (output > 0) *1.0
    output = output @ per[8]
    output *= (output > 0) *1.0
    output = output * actions + actions
    action = np.argmax(output)
    return action, per