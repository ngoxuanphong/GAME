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
    return [np.zeros((getActionSize() ** 2, getActionSize())),
        np.argsort(np.argsort(np.random.rand(getActionSize() ** 2, getActionSize()),axis = 1),axis = 1)*1.0,
        np.zeros((1,2))-1.]
@njit()
def argSortSpecial(w):
    indexTable = np.empty_like(w)
    indexTable2 = np.empty_like(w)
    for j in range(indexTable.shape[0]):
        indexTable[j,:] = np.argsort(w[j,:])
    for k in range(indexTable.shape[0]):
        indexTable2[k,:] = np.argsort(indexTable[k,:])
    return indexTable2 * 1.0

@njit()
def Train(state,per):
    if per[2][0][0]==-1 or per[2][0][1]==-1:
        action = np.random.choice(getValidActions(state))
        if per[2][0][0]==-1:
            per[2][0][0] = action
        elif per[2][0][1]==-1:
            per[2][0][1] = action
    else:
        weight = per[1][int(per[2][0][0] * getActionSize() + per[2][0][1])]
        action = np.argmax(weight*getValidActions(state))
        per[2][0][0] = per[2][0][1]
        per[2][0][1] = action
    
    if getReward(state)!=-1:
        if getReward(state)==1:
            per[0]+=per[1] * 1.0
        else:
            per[1] = argSortSpecial(np.random.rand(getActionSize() ** 2, getActionSize())) * 1.0
    return int(action),per
    
@njit()
def Test(state,per):
    if per[1][0][0]==-1 or per[1][0][1]==-1:
        action = np.random.choice(getValidActions(state))
        if per[1][0][0]==-1:
            per[1][0][0] = action
        elif per[1][0][1]==-1:
            per[1][0][1] = action
    else:
        weight = per[0][int(per[1][0][0] * getActionSize() + per[1][0][1])]
        action = np.argmax(weight*getValidActions(state))
        per[1][0][0] = per[1][0][1]
        per[1][0][1] = action
    return int(action),per
    
