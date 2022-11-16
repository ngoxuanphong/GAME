

# import tensorflow as tf
import numpy as np
import random as rd
from numba import njit
from numba import cuda

import os
import sys
from setup import game_name,time_run_game
sys.path.append(os.path.abspath(f"base/{game_name}"))
from env import *

player = 'NhatAnh_270922'
path_data = f'Agent/{player}/Data'
if not os.path.exists(path_data):
    os.mkdir(path_data)
path_save_player = f'Agent/{player}/Data/{game_name}_{time_run_game}/'
if not os.path.exists(path_save_player):
    os.mkdir(path_save_player)

def random_p(state,file_temp,file_per):
    list_action = getValidActions(state)
    list_action = np.where(list_action == 1)[0]
    ind = np.random.randint(len(list_action))
    action = list_action[ind]
    if getReward(state) == 1:
        np.save(f'{path_save_player}state.npy',state)
    return action,file_temp,file_per

@njit()
def basic_act(state,base):
    actions = getValidActions(state)
    actions = np.where(actions == 1)[0]
    for act in base:
        if act in actions:
            return act
    ind = np.random.randint(len(actions))
    action = actions[ind]
    return action

def create_per():
    new = np.arange(getActionSize())
    np.random.shuffle(new)
    return new

def rank(x):
    kq = x.copy()
    me = kq[0]
    kq.sort(reverse=True)
    return kq.index(me)

def test(state,temp,per):
    if len(temp) < 2:
        player = 'NhatAnh_270922'
        path_save_player = f'Agent/{player}/Data/{game_name}_{time_run_game}/'
        temp = np.load(f'{path_save_player}best.npy',allow_pickle=True)
    action = advance_act(state,temp)
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

def reshf(kq,per):
    try:
        per = [x for _, x in sorted(zip(kq, per),reverse=True)]
        return per
    except:
        return per
    

@njit()
def create_data(state):
    weight = np.random.randint(-1,2,size=(len(state),1))
    return weight

def advance_act(state,data):
    for id in range(len(data[1])):
        mt = np.dot(state,data[1][id])
        if mt[0] <= 0:
            action = basic_act(state,data[0][id-1])
            return action
        else:
            action = basic_act(state,data[0][id])
    return action

def create_per1(form,state):
    form[1][-1] = np.random.randint(-1,2,size=(len(state),1))
    return form

def ad0(state,temp,per):
    action = advance_act(state,per[0])
    return action,temp,per
def ad1(state,temp,per):
    action = advance_act(state,per[1])
    return action,temp,per
def ad2(state,temp,per):
    action = advance_act(state,per[2])
    return action,temp,per
def ad3(state,temp,per):
    action = advance_act(state,per[3])
    return action,temp,per
def ad4(state,temp,per):
    action = advance_act(state,per[4])
    return action,temp,per

def create_per_last(form,new,state):
    form[0].append(new)
    form[1].append(np.random.randint(-1,2,size=(len(state),1)))
    return form

# step 1
def first_bias():
    lp = [random_p] * getAgentSize()
    kq,file = normal_main(lp,1,[0])
    per = []
    for _ in range(getAgentSize()):
        per.append(create_per())
    pools = [test0,test1,test2,test3,test4]
    lp = pools[:getAgentSize()]
    count = 0
    for a in range(1000):
        per = [create_per()] + per
        kq,_ = normal_main(lp,10,per)
        if rank(kq) == 0:
            kq,_ = normal_main(lp,100,per)
            if rank(kq) == 0:
                kq,_ = normal_main(lp,1000,per)
                if rank(kq) == 0:
                    kq,_ = normal_main(lp,10000,per)
                    ranking = rank(kq)
                    per = reshf(kq,per[:getAgentSize()])
        else:
            per = per[1:]
    return per


# step 2
def step2(per,state):
    weights = [np.ones((len(state),1)),create_data(state)]
    new = ([per[:2],weights])
    per = [new] + per
    return per

#step 3
def step3(per,state):
    pools = [ad0,test1,test2,test3,test4]
    lp1 = pools[:getAgentSize()]
    count = 0
    max = 0
    layer0 = []
    a = 0
    while len(layer0) < getAgentSize()-1:
        new = [create_per1(per[0],state)]
        per = new + per[1:]
        kq,_ = normal_main(lp1,10,per)
        if np.argmax(kq) == 0 and kq[0]*1000 > max:
            kq,_ = normal_main(lp1,100,per)
            if np.argmax(kq) == 0 and kq[0]*100 > max:
                kq,_ = normal_main(lp1,1000,per)
                if np.argmax(kq) == 0 and kq[0]*10 > max:
                    kq,_ = normal_main(lp1,10000,per)
                    if np.argmax(kq) == 0 and kq[0] > max:
                        layer0.append(per[0])
                        np.save(f'{path_save_player}best.npy',per[-1])
                        max = kq[0]
    return layer0
def step4(per):
    per = [create_per()] + per[-(getAgentSize()-1):]
    pools = [test0,ad1,ad2,ad3,ad4]
    lp1 = pools[:getAgentSize()]
    max = 0
    new_layer = None
    for a in range(1000):
        per = [create_per()] + per[1:]
        kq,_ = normal_main(lp1,10,per)
        if kq[0]*1000 > max:
            kq,_ = normal_main(lp1,100,per)
            if kq[0]*100 > max:
                kq,_ = normal_main(lp1,1000,per)
                if kq[0]*10 > max:
                    kq,_ = normal_main(lp1,10000,per)
                    if kq[0] > max:
                        new_layer = per[0]
                        max = kq[0]
    return new_layer

def step5(per,new,state):
    per1 = [create_per_last(per[-1],new,state)] + per
    pools = [ad0,ad1,ad2,ad3,ad4]
    lp1 = pools[:getAgentSize()]
    max = 0
    layer0 = []
    while len(layer0) < getAgentSize()-1:
        new1 = [create_per_last(per[-1],new,state)]
        per1 = new1 + per1[1:]
        kq,_ = normal_main(lp1,10,per1)
        if np.argmax(kq) == 0 and kq[0]*1000 > max:
            kq,_ = normal_main(lp1,100,per1)
            if np.argmax(kq) == 0 and kq[0]*100 > max:
                kq,_ = normal_main(lp1,1000,per1)
                if np.argmax(kq) == 0 and kq[0]*10 > max:
                    kq,_ = normal_main(lp1,10000,per1)
                    if np.argmax(kq) == 0 and kq[0] > max:
                        layer0.append(per1[0])
                        np.save(f'{path_save_player}best.npy',per1[-1])
                        max = kq[0]
    return layer0

def train(n):
    per = first_bias()
    state = np.load(f'{path_save_player}state.npy',allow_pickle=True)
    per = step2(per,state)
    per = step3(per,state)
    np.save(f'{path_save_player}best.npy',per[-1])
    for xy in range(n-1):
        new = step4(per)
        per = step5(per,new,state)

