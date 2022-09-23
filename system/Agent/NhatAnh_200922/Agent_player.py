import random as rd
import numpy as np
from numba import njit
from numba import cuda
# import tensorflow as tf

import os
import sys
from setup import game_name,time_run_game
sys.path.append(os.path.abspath(f"base/{game_name}"))
from env import *

player = 'NhatAnh_200922'
path_data = f'system/Agent/{player}/Data'
if not os.path.exists(path_data):
    os.mkdir(path_data)
path_save_player = f'system/Agent/{player}/Data/{game_name}_{time_run_game}/'
if not os.path.exists(path_save_player):
    os.mkdir(path_save_player)

def random_p(state,file_temp,file_per):
    list_action = get_list_action(state)
    ind = np.random.randint(len(list_action))
    action = list_action[ind]
    return action,file_temp,file_per

@njit()
def basic_act(state,base):
    actions = get_list_action(state)
    for act in base:
        if act in actions:
            return act
    ind = np.random.randint(len(actions))
    action = actions[ind]
    return action

def create0(state,temp,per):
    if len(per) < 2:
        per = [[],0]
    if len(temp) < 2:
        temp = np.arange(amount_action())
        np.random.shuffle(temp)
    action = basic_act(state,temp)
    if check_victory(state) == 1:
        per[0].append(temp)
    return action,temp,per

def test(state,temp,per):
    if len(temp) < 2:
        player = 'NhatAnh_200922'
        path_save_player = f'system/Agent/{player}/Data/{game_name}_{time_run_game}/'
        temp = np.load(f'{path_save_player}best.npy',allow_pickle=True)
    action = basic_act(state,temp)
    return action,temp,per

def test_1(state,temp,per):
    if len(temp) < 2:
        temp = np.load(f'{path_save_player}best1.npy',allow_pickle=True)
    action = basic_act(state,temp)
    return action,temp,per
def test_2(state,temp,per):
    if len(temp) < 2:
        temp = np.load(f'{path_save_player}best2.npy',allow_pickle=True)
    action = basic_act(state,temp)
    return action,temp,per
def test_3(state,temp,per):
    if len(temp) < 2:
        temp = np.load(f'{path_save_player}best3.npy',allow_pickle=True)
    action = basic_act(state,temp)
    return action,temp,per
def test_4(state,temp,per):
    if len(temp) < 2:
        temp = np.load(f'{path_save_player}best4.npy',allow_pickle=True)
    action = basic_act(state,temp)
    return action,temp,per

def test0(state,temp,per):
    action = basic_act(state,per[0])
    return action,temp,per
def test1(state,temp,per):
    action = basic_act(state,per[1])
    return action,temp,per
def test2(state,temp,per):
    action = basic_act(state,per[2])
    return action,temp,per
def test3(state,temp,per):
    action = basic_act(state,per[3])
    return action,temp,per
def test4(state,temp,per):
    action = basic_act(state,per[4])
    return action,temp,per


# tạo con mạnh nhất đầu tiên
def step1(ten):
    lp = [create0] * amount_player()
    kq,file = normal_main(lp,amount_player()**2,[0])
    per = file[0]
    # print(len(per))
    pools = [test0,test1,test2,test3,test4]
    lp = pools[:amount_player()]
    kq,_ = normal_main(lp,1000,per)
    best = per[np.argmax(kq)]
    np.save(ten,best)

# lọc từ 1 popu ra 1 con mạnh nhất
def screen0(popu,ten):
    step = len(popu)//(amount_player() -1)
    amount = amount_player() -1 
    for game in range(step):
        per = popu[game*amount:(game+1)*amount]
        pools = [test,test0,test1,test2,test3,test4]
        lp = pools[:amount_player()]
        try:
            kq,_ = normal_main(lp,amount_player()* (amount_player()+1) + 1,per)
            # kq,_ = normal_main(lp,1,per)
            idb = np.argmax(kq)
            if idb != 0:
                best = per[idb-1]
                np.save(ten,best)
        except:
            continue
    remain = len(popu)%(amount)
    if remain > 0:
        per = popu[-remain:]
        per += popu[:(amount-remain)]
        pools = [test,test0,test1,test2,test3,test4]
        lp = pools[:amount_player()]
        try:
            kq,_ = normal_main(lp,amount_player()* (amount_player()+1) + 1,per)
            # kq,_ = normal_main(lp,1,per)
            idb = np.argmax(kq)
            if idb != 0:
                best = per[idb-1]
                np.save(ten,best)
        except:
            pass

# sắp xếp thứ tự action
def step2(ten):
    for id in range(amount_action()):
        # print(id)
        best = list(np.load(ten,allow_pickle=True))
        popu = []
        for act in range(amount_action()):
            new = best[:id] + [act] + best[id:]
            popu.append(new)
        screen0(popu,ten)
    temp = np.load(ten,allow_pickle=True)
    best = []
    for x in temp:
        if x not in best:
            best.append(x)
    np.save(ten,best)

def step3():
    for id in range(1,amount_player()):
        ten = path_save_player + "best" + str(id) + ".npy"
        step1(ten)
        step2(ten)
    pools = [test,test_1,test_2,test_3,test_4]
    lp = pools[:amount_player()]
    kq,_ = normal_main(lp,10000,[0])
    idb = np.argmax(kq)
    if idb != 0:
        ten = path_save_player+  "best" + str(idb) + ".npy"
        best = np.load(ten,allow_pickle=True)
        np.save(f'{path_save_player}best.npy',best)

def train(n):
    step1(f'{path_save_player}best.npy')
    step2(f'{path_save_player}best.npy')
    for _ in range(n-1):
        step3()
    return np.load(f'{path_save_player}best.npy',allow_pickle=True)