import numpy as np
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

def DataAgent():
    return [np.random.choice(np.arange(env.getActionSize()),size=env.getActionSize(),replace=False) * 1.0,
    np.zeros(env.getActionSize()),
    np.zeros(10) #2: 0 là mode (0 - train, 1 - test, 2 - per) , 1 là số trận đã thắng (data), 2 là tỉ lệ test thắng max, 3 là số lần không vượt max, 4 là số ván chơi test, 5 là số ván win test
    ]

@njit()
def Train(state,per):
    actions = env.getValidActions(state)
    win = env.getReward(state)
    mode = per[2][0]
    if mode == 0:
        output = actions * per[0] + actions
        action = np.argmax(output)        
        if win == 1:
            per[1] += per[0]
            per[2][1] += 1
            if per[2][1] % 1000 == 0:
                per[2][0] = 1
            # per[0] = np.random.choice(np.arange(env.getActionSize()),size=env.getActionSize(),replace=False) * 1.0
        if win == 0:
            per[0] = np.random.choice(np.arange(env.getActionSize()),size=env.getActionSize(),replace=False) * 1.0
    if mode == 1:
        bias = per[1]/np.max(per[1])
        output = actions * bias + actions
        action = np.argmax(output)
        if win != -1:
            per[2][4] += 1
            if win == 1:
                per[2][5] += 1
            if per[2][4] == 1000000:
                win_rate = per[2][5]/per[2][4]
                per[2][5] = 0
                per[2][4] = 0
                if win_rate > per[2][2]:
                    per[2][2] = win_rate
                    per[2][3] = 0
                else:
                    per[2][3] += 1
                    # print("1 lần không vượt, tổng là", per[2][3])
                if per[2][3] == 3:
                    per[2][0] = 2
                else:
                    per[2][0] = 0
    if mode == 2:
        bias = per[1]/np.max(per[1])
        output = actions * bias + actions
        action = np.argmax(output)
    return action,per



@njit()
def Test(state,per):
    actions = env.getValidActions(state)
    win = env.getReward(state)
    mode = per[2][0]
    if mode == 0:
        output = actions * per[0] + actions
        action = np.argmax(output)        
        if win == 1:
            per[1] += per[0]
            per[2][1] += 1
            if per[2][1] % 1000 == 0:
                per[2][0] = 1
            # per[0] = np.random.choice(np.arange(env.getActionSize()),size=env.getActionSize(),replace=False) * 1.0
        if win == 0:
            per[0] = np.random.choice(np.arange(env.getActionSize()),size=env.getActionSize(),replace=False) * 1.0
    if mode == 1:
        bias = per[1]/np.max(per[1])
        output = actions * bias + actions
        action = np.argmax(output)
        if win != -1:
            per[2][4] += 1
            if win == 1:
                per[2][5] += 1
            if per[2][4] == 1000000:
                win_rate = per[2][5]/per[2][4]
                per[2][5] = 0
                per[2][4] = 0
                if win_rate > per[2][2]:
                    per[2][2] = win_rate
                    per[2][3] = 0
                else:
                    per[2][3] += 1
                    # print("1 lần không vượt, tổng là", per[2][3])
                if per[2][3] == 3:
                    per[2][0] = 2
                else:
                    per[2][0] = 0
    if mode == 2:
        bias = per[1]/np.max(per[1])
        output = actions * bias + actions
        action = np.argmax(output)
    return action,per

