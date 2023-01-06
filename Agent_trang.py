import numpy as np
import random as rd
from numba import njit, jit
from numba.typed import List
import sys, os
from setup import SHOT_PATH
import importlib.util
game_name = 'Splendor_v2'

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

from base.TLMN.env import *


@njit
def Train(state, per):

    if per[4][0][0] < 10000:
        if per[3][0][0] == -1:
            choice = np.argmin(per[2][0])
            per[3][0][0] = choice
            per[2][0][choice] += 1

        choice_id = int(per[3][0][0])
        bias = per[0][choice_id]
        actions = getValidActions(state)
        kq = actions * bias
        action = np.argmax(kq)

        win = getReward(state)
        if win != -1:
            if win == 1:
                per[1][0][choice_id] += 1

            per[4][0][0] += 1
            if per[4][0][0] == 10000:
                win_rate = per[1][0] / per[2][0]
                best = np.argmax(win_rate)
                per[0][0] = per[0][best]
                per[3][0][0] = win_rate[best]
                per[2][0] = 0
                per[1][0] = 0
            else:
                per[3][0][0] = -1

        return action, per

    if per[4][0][0] < 10999:
        actions = getValidActions(state)
        kq = actions * per[0][0]
        action = np.argmax(kq)
        list_action = np.where(actions == 1)[0]
        if len(list_action) > 1:
            per[6][0][action] += 1
            for a in list_action:
                per[5][a][list_action] += 1
                per[5][a][a] = 0

        win = getReward(state)
        if win != -1:
            id_match = int(per[4][0][0]) % 1000
            per[4][0][0] += 1
            per[7][0][id_match] = win
            if per[4][0][0] == 10999:
                win_rate = np.sum(per[7][0]) / 100
                if win_rate > per[3][0][0]:
                    per[3][0][0] = win_rate

        return action, per

    if per[2][0][99] == 0:
        actions = getValidActions(state)
        kq = actions * per[0][0]
        action = np.argmax(kq)
        list_action = np.where(actions == 1)[0]
        if len(list_action) > 1:
            per[6][0][action] += 1
            for a in list_action:
                per[5][a][list_action] += 1
                per[5][a][a] = 0

        win = getReward(state)
        if win != -1:
            id_match = int(per[4][0][0]) % 1000
            per[4][0][0] += 1
            per[7][0][id_match] = win
            win_rate = np.sum(per[7][0]) / 100
            if win_rate > per[3][0][0]:
                per[3][0][0] = win_rate
            else:
                if int(per[4][0][0]) % 3000 == 0:
                    num_1 = np.count_nonzero(per[6][0] != 0)
                    check = False
                    for l_i in range(num_1):
                        action_max = np.argmax(per[6][0])
                        per[6][0][action_max] = 0
                        aptgt = per[5][action_max].copy()
                        num_2 = np.count_nonzero(aptgt != 0)
                        for l_j in range(num_2):
                            action_choice = np.argmax(aptgt)
                            aptgt[action_choice] = 0
                            if per[8][action_max][action_choice] == 0:
                                per[8][action_max][action_choice] = 1
                                per[8][action_choice][action_max] = 1
                                per[0][1] = per[0][0]
                                temp_s = per[0][1][action_max]
                                per[0][1][action_max] = per[0][1][action_choice]
                                per[0][1][action_choice] = temp_s

                                check = True
                                break

                        if check:
                            break

                    if check:
                        per[2][0][99] = 1
                        per[1][0][0:2] = 0
                        per[2][0][0:2] = 0

        return action, per

    if per[2][0][99] == 1:
        if per[2][0][0] < 2000:
            actions = getValidActions(state)
            kq = actions * per[0][0]
            action = np.argmax(kq)

            win = getReward(state)
            if win != -1:
                per[2][0][0] += 1
                if win == 1:
                    per[1][0][0] += 1

        else:
            actions = getValidActions(state)
            kq = actions * per[0][1]
            action = np.argmax(kq)

            win = getReward(state)
            if win != -1:
                per[2][0][1] += 1
                if win == 1:
                    per[1][0][1] += 1

                if per[2][0][1] == 500:
                    num_1 = per[1][0][0] / per[2][0][0]
                    num_2 = per[1][0][1] / per[2][0][1]

                    if num_2 > num_1:
                        per[0][0] = per[0][1]

                    per[2][0][99] = 0

                    per[6] = np.zeros((1, getActionSize()))
                    per[5] = np.zeros((getActionSize(), getActionSize()))

        return action, per

@njit
def Test(state, per):

    if per[4][0][0] < 10000:
        if per[3][0][0] == -1:
            choice = np.argmin(per[2][0])
            per[3][0][0] = choice
            per[2][0][choice] += 1

        choice_id = int(per[3][0][0])
        bias = per[0][choice_id]
        actions = getValidActions(state)
        kq = actions * bias
        action = np.argmax(kq)

        win = getReward(state)
        if win != -1:
            if win == 1:
                per[1][0][choice_id] += 1

            per[4][0][0] += 1
            if per[4][0][0] == 10000:
                win_rate = per[1][0] / per[2][0]
                best = np.argmax(win_rate)
                per[0][0] = per[0][best]
                per[3][0][0] = win_rate[best]
                per[2][0] = 0
                per[1][0] = 0
            else:
                per[3][0][0] = -1

        return action, per

    if per[4][0][0] < 10999:
        actions = getValidActions(state)
        kq = actions * per[0][0]
        action = np.argmax(kq)
        list_action = np.where(actions == 1)[0]
        if len(list_action) > 1:
            per[6][0][action] += 1
            for a in list_action:
                per[5][a][list_action] += 1
                per[5][a][a] = 0

        win = getReward(state)
        if win != -1:
            id_match = int(per[4][0][0]) % 1000
            per[4][0][0] += 1
            per[7][0][id_match] = win
            if per[4][0][0] == 10999:
                win_rate = np.sum(per[7][0]) / 100
                if win_rate > per[3][0][0]:
                    per[3][0][0] = win_rate

        return action, per

    if per[2][0][99] == 0:
        actions = getValidActions(state)
        kq = actions * per[0][0]
        action = np.argmax(kq)
        list_action = np.where(actions == 1)[0]
        if len(list_action) > 1:
            per[6][0][action] += 1
            for a in list_action:
                per[5][a][list_action] += 1
                per[5][a][a] = 0

        win = getReward(state)
        if win != -1:
            id_match = int(per[4][0][0]) % 1000
            per[4][0][0] += 1
            per[7][0][id_match] = win
            win_rate = np.sum(per[7][0]) / 100
            if win_rate > per[3][0][0]:
                per[3][0][0] = win_rate
            else:
                if int(per[4][0][0]) % 3000 == 0:
                    num_1 = np.count_nonzero(per[6][0] != 0)
                    check = False
                    for l_i in range(num_1):
                        action_max = np.argmax(per[6][0])
                        per[6][0][action_max] = 0
                        aptgt = per[5][action_max].copy()
                        num_2 = np.count_nonzero(aptgt != 0)
                        for l_j in range(num_2):
                            action_choice = np.argmax(aptgt)
                            aptgt[action_choice] = 0
                            if per[8][action_max][action_choice] == 0:
                                per[8][action_max][action_choice] = 1
                                per[8][action_choice][action_max] = 1
                                per[0][1] = per[0][0]
                                temp_s = per[0][1][action_max]
                                per[0][1][action_max] = per[0][1][action_choice]
                                per[0][1][action_choice] = temp_s

                                check = True
                                break

                        if check:
                            break

                    if check:
                        per[2][0][99] = 1
                        per[1][0][0:2] = 0
                        per[2][0][0:2] = 0

        return action, per

    if per[2][0][99] == 1:
        if per[2][0][0] < 2000:
            actions = getValidActions(state)
            kq = actions * per[0][0]
            action = np.argmax(kq)

            win = getReward(state)
            if win != -1:
                per[2][0][0] += 1
                if win == 1:
                    per[1][0][0] += 1

        else:
            actions = getValidActions(state)
            kq = actions * per[0][1]
            action = np.argmax(kq)

            win = getReward(state)
            if win != -1:
                per[2][0][1] += 1
                if win == 1:
                    per[1][0][1] += 1

                if per[2][0][1] == 500:
                    num_1 = per[1][0][0] / per[2][0][0]
                    num_2 = per[1][0][1] / per[2][0][1]

                    if num_2 > num_1:
                        per[0][0] = per[0][1]

                    per[2][0][99] = 0

                    per[6] = np.zeros((1, getActionSize()))
                    per[5] = np.zeros((getActionSize(), getActionSize()))

        return action, per



@njit()
def DataAgent():
    per_Ann = List()
    per_Ann.append(np.random.rand(100, getActionSize()))  # 0
    per_Ann.append(np.zeros((1, 100)))  # 1
    per_Ann.append(np.zeros((1, 100)))  # 2
    per_Ann.append(np.array([[-1.]]))  # 3
    per_Ann.append(np.array([[0.]]))  # 4
    per_Ann.append(np.zeros((getActionSize(), getActionSize())))  # 5
    per_Ann.append(np.zeros((1, getActionSize())))  # 6
    per_Ann.append(np.zeros((1, 1000)))  # 7
    per_Ann.append(np.zeros((getActionSize(), getActionSize())))  # 8
    return per_Ann
    
import time

a = time.time()
per = DataAgent()
win, per = numba_main_2(Train, 10000, per, 0)
print('train', win)

# win, per = numba_main_2(Test, 10000, per, 0)
b = time.time()
print('test', win, b - a)