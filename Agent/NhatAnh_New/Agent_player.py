import random as rd
import numpy as np

import os
import sys
from setup import game_name,time_run_game
sys.path.append(os.path.abspath(f"base/{game_name}"))
from env import *

player = 'NhatAnh_New'
path_data = f'Agent/{player}/Data'
if not os.path.exists(path_data):
    os.mkdir(path_data)
path_save_player = f'Agent/{player}/Data/{game_name}_{time_run_game}/'
if not os.path.exists(path_save_player):
    os.mkdir(path_save_player)

#@title Shuffle
def shuffle(state,temp,per):
    if len(temp) < 2:
        temp = np.arange(amount_action())
        rd.shuffle(temp)
    actions = get_list_action(state)
    action = None
    max = -1
    for act in actions:
        score = temp[act]
        if score > max:
            max = score
            action = act
    if check_victory(state) == 1:
        if len(per) < 2:
            try: 
                best = np.load(f'{path_save_player}best.npy',allow_pickle=True)
                per = [best]
            except:
                per = [temp]
        per.append(temp)
    return action, temp,per

def test_shuffle(state,temp,per):
    if len(per) < 2:
        popu = np.load(f'{path_save_player}population.npy',allow_pickle=True)
        per = [popu,np.zeros(len(popu)),np.zeros(len(popu))]
    if len(temp) < 2:
        a = per[1]
        ind = np.random.choice(np.where(a == np.min(a))[0])
        temp = [per[0][ind],ind]
        # cộng 1 vào số lần chơi của mt
        per[1][ind] += 1
    actions = get_list_action(state)
    action = None
    max = -1
    for act in actions:
        score = temp[0][act]
        if score > max:
            max = score
            action = act
    if check_victory(state) == 1:
        per[2][temp[1]] += 1
    return action, temp,per

def test(state,temp,per):
    player = 'NhatAnh_New'
    path_save_player = f'Agent/{player}/Data/{game_name}_{time_run_game}/'
    if len(temp) < 2:
        temp = [np.load(f'{path_save_player}best.npy',allow_pickle=True),[0,0]]
    actions = get_list_action(state)
    action = None
    max = -1
    model = temp[0]
    for act in actions:
        score = model[act]
        if score > max:
            max = score
            action = act
    return action, temp,per


def train(n):
    for _ in range(n):
        list_player = [shuffle]*amount_player()
        kq,file = normal_main(list_player,10000,[0])
        np.save(f'{path_save_player}population.npy',file)
        list_player = [test_shuffle]*amount_player()
        kq,file = normal_main(list_player,20000,[0])
        result = file[2]/file[1]
        best = file[0][np.argmax(result)]
        np.save(f'{path_save_player}best.npy',best)
    return best