from numba import cuda
from numba import njit
import tensorflow as tf
from env import *
import random as rd
import numpy as np

# print('hhhh')
import os
import sys
from setup import game_name, time_run_game
sys.path.append(os.path.abspath(f"base/{game_name}"))
player = 'NhatAnh_130922'
path_data = f'Agent/{player}/Data'
if not os.path.exists(path_data):
    os.mkdir(path_data)
path_save_player = f'Agent/{player}/Data/{game_name}_{time_run_game}/'
if not os.path.exists(path_save_player):
    os.mkdir(path_save_player)


# @title Shuffle
def random_p(state, file_temp, file_per):
    list_action = getValidActions(state)
    list_action = np.where(list_action == 1)[0]
    action = np.random.choice(list_action)
    return action, file_temp, file_per


@njit()
def create_data(state, size):
    networks = np.random.randint(1, 5, size=size)
    weights = []
    space = len(state)
    if size == 0:
        weight = np.random.randint(0, 2, size=(space, 1))
        weights.append(weight)
    else:
        for ind in range(len(networks)):
            if ind == 0:
                weight = np.random.randint(0, 2, size=(space, networks[ind]))
            else:
                weight = np.random.randint(
                    0, 2, size=(networks[ind-1], networks[ind]))
            weights.append(weight)
        weight = np.random.randint(0, 2, size=(networks[-1], 1))
        weights.append(weight)
    layer = np.zeros(getActionSize())
    action = np.random.randint(getActionSize())
    layer[action] = 1
    return (weights, layer)


def data_to_layer(state, data):
    for ind in data[0]:
        state = np.dot(state, ind)
        state *= state > 0
    active = state > 0
    layer = data[1] * active
    return layer


def test(state, temp, per):
    player = 'NhatAnh_130922'
    path_save_player = f'Agent/{player}/Data/{game_name}_{time_run_game}/'
    if len(temp) < 2:
        best = np.load(f'{path_save_player}best.npy', allow_pickle=True)
        temp = [list(best), 0]
    layer = np.zeros(getActionSize())
    for data in temp[0]:
        layer += data_to_layer(state, data)
    base = np.zeros(getActionSize())
    actions = getValidActions(state)
    actions = np.where(actions == 1)[0]
    for act in actions:
        base[act] = 1
    layer *= base
    base += layer
    action = np.random.choice(np.where(base == np.max(base))[0])
    return action, temp, per


def create(state, temp, per):
    # state = 1.0 * state
    if len(per) < 2:
        popu = []  # index 0
        time = 0  # index 1
        win = 0  # index 2
        best = np.load(f'{path_save_player}best.npy',
                       allow_pickle=True)  # index 3
        mode = "create"  # index 4
        per = [popu, time, win, best, mode]
    if len(temp) < 2:
        if per[4] == "create":
            temp = [create_data(state, 0), "create"]
        else:
            if per[4] == "compete":
                a = per[1]
                ind = np.random.choice(np.where(a == np.min(a))[0])
                temp = [per[0][ind], ind]
                per[1][ind] += 1
    base = np.zeros(getActionSize())
    actions = getValidActions(state)
    actions = np.where(actions == 1)[0]
    for act in actions:
        base[act] = 1
    layer = data_to_layer(state, temp[0])
    for data in per[3]:
        layer += data_to_layer(state, data)
    layer *= base
    base += layer
    action = np.random.choice(np.where(base == np.max(base))[0])
    if getReward(state) == 1:
        if per[4] == "create":
            per[0].append(temp[0])
            if len(per[0]) == 100:
                per[1] = np.zeros(100)
                per[2] = np.zeros(100)
                per[4] = "compete"
        else:
            if per[4] == "compete":
                per[2][temp[1]] += 1
    return action, temp, per


def creating():
    list_player = [create]*(getAgentSize()-1) + [random_p]
    kq, file = normal_main(list_player, 1000, [0])
    check = file[0][np.argmax(file[2]/file[1])]
    np.save(f'{path_save_player}check.npy', check)
    return check


def check(state, temp, per):
    if len(per) < 2:
        check = np.load(f'{path_save_player}check.npy', allow_pickle=True)
        best = np.load(f'{path_save_player}best.npy', allow_pickle=True)
        appearance = 0
        wins = 0
        per = [check, best, appearance, wins]
    base = np.zeros(getActionSize())
    actions = getValidActions(state)
    actions = np.where(actions == 1)[0]
    for act in actions:
        base[act] = 1
    layer = data_to_layer(state, per[0])
    if np.max(layer*base) > 0:
        per[2] = 1
    for data in per[1]:
        layer += data_to_layer(state, data)
    layer *= base
    base += layer
    action = np.random.choice(np.where(base == np.max(base))[0])
    win = getReward(state)
    if win != -1:
        if per[2] == 1:
            per[3] += win
        per[2] = 0
    return action, temp, per


def checking(check):
    list_player = [check, check, test, test] + [random_p] * (getAgentSize()-4)
    kq, file = normal_main(list_player, 1000, [0])
    win = file[3]
    rate = (kq[0] + kq[1]) - (kq[2] + kq[3])
    if rate > 0 and win > rate:
        check = np.load(f'{path_save_player}check.npy', allow_pickle=True)
        best = np.load(f'{path_save_player}best.npy', allow_pickle=True)
        best = list(best)
        best.append(check)
        np.save(f'{path_save_player}best.npy', best)
    return win, kq


def train(n):
    best = []
    np.save(f'{path_save_player}best.npy', best)
    for _ in range(10*n):
        creating()
        checking(check)
