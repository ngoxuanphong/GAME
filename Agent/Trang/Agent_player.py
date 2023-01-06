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

@njit()
def Test(p_state, per):
    actions = getValidActions(p_state)
    actions = np.where(actions == 1)[0]
    Result_matran1 = np.dot(p_state, per[6][0])
    max_val_act = -9999
    for act in actions:
        if Result_matran1[int(act)] > max_val_act:
            action_max = int(act)
            max_val_act = Result_matran1[int(act)]

    return action_max, per

@njit()
def Train(p_state, per):
    actions = getValidActions(p_state)
    actions = np.where(actions == 1)[0]

    if per[2][0][0][0] == 0:
        id_choose = np.argmin(per[1][0][0])
        per[1][0][0][id_choose] += 1
        per[3][0][0][0] = id_choose
        per[2][0][0][0] = 1
    else:
        id_choose = int(per[3][0][0][0])

    Result_matran1 = np.dot(p_state, per[0][id_choose])
    max_val_act = -9999
    for act in actions:
        if Result_matran1[int(act)] > max_val_act:
            action_max = int(act)
            max_val_act = Result_matran1[int(act)]

    check_win = getReward(p_state)
    if check_win == 1:
        per[4][0][0][id_choose] = per[4][0][0][id_choose] + 1.2
    if check_win == 0:
        per[4][0][0][id_choose] -= 0.8

    if check_win != -1:
        per[2][0][0][0] = 0
        per[5][0][0][0] += 1

        if per[5][0][0][0] == 3000:
            id_max = np.argmax(per[4][0][0])
            per[6][0] = per[0][id_max]
            per[0] = np.random.random((100, getStateSize(),getActionSize()))*2 - 0.6
            per[0][0] = per[6][0]

            per[1] = np.array([[[0. for i in range(100)]]])
            per[4] = np.array([[[0. for i in range(100)]]])
            per[5] = np.array([[[0.]]])
    return action_max, per

@njit()
def DataAgent():
    per0 = np.random.random((100, getStateSize(),getActionSize()))*2 - 0.6
    per1 = np.array([[[0. for i in range(100)]]]) #So tran da train cua moi ma tran
    per2 = np.array([[[0.]]]) #Heets game hay chua
    per3 = np.array([[[0.]]]) #Id ma tran dang chon de dung
    per4 = np.array([[[0. for i in range(100)]]]) #Diem cua cac ma tran
    per5 = np.array([[[0.]]]) #So tran da train
    per6 = np.random.random((1, getStateSize(),getActionSize()))*2 - 0.6
    per = [per0, per1, per2, per3, per4, per5, per6]
    return per