# Value Greedy
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
    per_find_value = [np.zeros(2), #[0][0] số trận lấy min max, [0][1]: 1 la da tim duoc value
                    np.zeros(getStateSize()), # luu value o day
                    np.zeros(getStateSize()), # [2]: max, state cũ
                    np.zeros(getStateSize()), # [3]: min, action đã act ( -1 là k có action nào)
                    np.zeros(getActionSize()) # [4]: điểm từng action
                        ]
    return per_find_value

@njit()
def Train(state,per):
    actions = getValidActions(state)
    win = getReward(state)
    output = np.random.rand(getActionSize()) * actions + actions
    action = np.argmax(output)
    if per[0][1] == 0:
        if per[0][0] < 1000:
            if win == 1:
                per[1] = np.minimum(per[1],state)
                per[2] = np.maximum(per[2],state)
                per[3] = state
                per[0][0] += 1
        else:
            value = np.zeros(getStateSize())
            for id in range(getStateSize()):
                sample = np.zeros(getStateSize()) + per[3]
                ifmin, ifmax = 0,0
                sample[id] = per[1][id]
                if getReward(sample) == 1:
                    ifmin = 1
                sample[id] = per[2][id]
                if getReward(sample) == 1:
                    ifmax = 1
                if ifmax > ifmin:
                    value[id] = 1
                if ifmin > ifmax:
                    value[id] = -1
            per[1] = value
            per[2] = np.zeros(getStateSize())
            per[3] = np.zeros(2000) - 1
            per[0][1] = 1
    else:
        # khi mới bắt đầu game
        if np.min(per[2]) == np.max(per[2]):
            per[2] = state
            per[3][0] = action
        else:
            # tính value
            value = (state - per[2]) * per[1]
            value *= value > 0
            score = np.sum(value)
            # khi + value
            if score > 0:
                amountActed = np.sum(per[3] != -1)
                smallS = score/(amountActed * 2)
                bigS = score/2
                for act in per[3]:
                    if act != -1:
                        actB = int(act)
                        per[4][actB] += smallS
                    else:
                        per[4][actB] += bigS
                        break
            per[2] = state
            for id in range(len(per[3])):
                if per[3][id] == -1:
                    per[3][id] = action
                    break
        if win != -1:
            per[2] = np.zeros(getStateSize())
            per[3] = np.zeros(2000) -1
    return action, per

@njit()
def Test(state,per):
    actions = getValidActions(state)
    output = np.argsort(np.argsort(per[4])) * actions + actions
    action = np.argmax(output)
    return action, per
