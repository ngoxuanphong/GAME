import random as rd
from threading import local
import numpy as np

import os
import sys
from setup import game_name,time_run_game
sys.path.append(os.path.abspath(f"base/{game_name}"))
from env import *

player = 'NhatAnh_0822'
path_data = f'Agent/{player}/Data'
if not os.path.exists(path_data):
    os.mkdir(path_data)
path_save_player = f'Agent/{player}/Data/{game_name}_{time_run_game}/'
if not os.path.exists(path_save_player):
    os.mkdir(path_save_player)


def random_p(state,file_temp,file_per):
    list_action = get_list_action(state)
    action = np.random.choice(list_action)
    return action,file_temp,file_per
list_player = [random_p]*amount_player()
kq, file = normal_main(list_player,1, [0])
def reset(per):
    best = [] # index 0
    testing = 0 # index 1
    played = np.zeros(amount_action()) # index 2
    winning = np.zeros(amount_action()) # index 3
    rating = np.zeros(amount_action()) # index 4
    timer = 0 # index 5
    firmness = per[6] + 1 # index 6
    per = [best,testing,played,winning,rating,0,firmness]
    return per

def test(state,temp,per):
    if len(temp) < 2:
        player = 'NhatAnh_0822'
        path_save_player = f'Agent/{player}/Data/{game_name}_{time_run_game}/'
        temp = np.load(f'{path_save_player}best.npy',allow_pickle=True)
    list_action = get_list_action(state)
    action = np.random.choice(list_action)
    for act in temp:
        if act in list_action:
            action = act
            break
    return action,temp,per

def find_best(state,temp,per):
    if len(per) < 2:
        best = [] # index 0
        testing = 0 # index 1
        played = np.zeros(amount_action()) # index 2
        winning = np.zeros(amount_action()) # index 3
        rating = np.zeros(amount_action()) # index 4
        timer = 0 # index 5
        firmness = 1 # index 6
        per = [best,testing,played,winning,rating,0,firmness]
    if len(temp) < 2:
        temp = [0,0]
    list_action = get_list_action(state)
    action = None
    if len(per[0]) > 0:
        for act in per[0]:
            if act in list_action:
                action = act
                # print("chọn action max là",action)
                break
    if action == None:
        target = per[1]
        if target in list_action:
            action = target
        else:
            action = np.random.choice(list_action)
        if action == target:
            temp[0] += 1
    if check_victory(state) != -1:
        per[5] += 1
        if temp[0] > 0:
            per[2][per[1]] += 1
            per[3][per[1]] += check_victory(state)
        a = per[5] > 50 and per[2][per[1]]/per[5] < 0.02
        if per[2][per[1]] == 10**per[6] or a == True :
            # print("action",per[1],"rating là",per[3][per[1]]/per[2][per[1]],"sau",per[5],"trận")
            per[5] = 0
            per[1] += 1
            while per[1] in per[0]:
                per[1] += 1
            if per[1] > amount_action()-1:
                per[1] = 0
                full_rate = per[3]/per[2]
                max = -1 
                top_id = None
                for id_act in range(len(full_rate)):
                    if id_act not in per[0]:
                        score = full_rate[id_act]
                        if score > max:
                            max = score
                            top_id = id_act
                if top_id == None:
                    # print("train xong rồi nhé",10**(per[6]))
                    np.save(f'{path_save_player}best.npy',per[0])
                    per = reset(per)
                else:
                    per[0].append(top_id)
                    # print(per[0],top_id)
                    per[2] = np.zeros(amount_action())
                    per[3] = np.zeros(amount_action())
    return action,temp,per

def train(n):
    list_player = [random_p,random_p] + [find_best] * (amount_player() - 2)
    kq, file = normal_main(list_player,10000*n, [0])
    return np.load(f'{path_save_player}best.npy',allow_pickle=True)
