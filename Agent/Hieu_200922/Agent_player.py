import math
import numpy as np
import heapq
# from scipy.stats.mstats import gmean, hmean
import random
# import torch.nn as nn
# import torch.nn.functional as F

random.seed(100)
import os
import sys
from setup import game_name,time_run_game
sys.path.append(os.path.abspath(f"base/{game_name}"))
from env import *

player = 'Hieu_200922'  #Tên folder của người chơi
path_data = f'Agent/{player}/Data'
if not os.path.exists(path_data):
    # print("folder")
    os.mkdir(path_data)
path_save_player = f'Agent/{player}/Data/{game_name}_{time_run_game}/'
if not os.path.exists(path_save_player):
    os.mkdir(path_save_player)

def agent(state,file_temp,file_per):
    list_action = getValidActions(state)
    list_action = np.where(list_action == 1)[0]
    action = np.random.choice(list_action)
    file_per = (len(state),getActionSize())
    return action,file_temp,file_per

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def leaky_relu(x, theta = 0.05):
    return np.where(x > 0, x, x * theta)

LEN_STATE,AMOUNT_ACTION = normal_main([agent]*getAgentSize(), 1, [0])[1]
all_matrix  = []
all_new_matrix = []
all_matrix_score = []
all_so_tran = []
all_ratio = []
trial_matrix = [0, 0, 0]
ratio_win = []

def random_player(player_state, file_temp, file_per):
    list_action = getValidActions(player_state)
    action = int(np.random.choice(list_action))
    return action, file_temp, file_per

def Hieu_player1(state, file_temp, file_per):
    list_action = getValidActions(state)
    if len(file_temp) < 2:
        file_temp = [0, 0, 0]
        raw1 = len(state)
        col1 = min(AMOUNT_ACTION, LEN_STATE)
        raw2 = col1
        col2 = LEN_STATE//getAgentSize()
        raw3 = col2
        col3 = getActionSize()
        file_temp[0] = np.random.rand(raw1, col1)*3-1.7
        file_temp[1] = np.random.rand(raw2, col2)*2-1
        file_temp[2] = np.random.rand(raw3, col3)
    action = neural_network(state, file_temp, list_action)
    if action not in list_action:
        raise Exception('action sai', list_action, action)
    if getReward(state) == 1:
        if type(file_per[0]) == int:
            file_per = [file_temp]
        else:
            # if len(file_per) % 2000 == 0:
            #     print(len(file_per))
            file_per.append(file_temp)
    return action, file_temp, file_per

def Hieu_player_2(state, file_temp, file_per):
    global all_matrix, all_matrix_score, all_so_tran
    list_action = getValidActions(state)
    if len(file_temp) < 2:
        id_matrix = np.random.randint(0, len(all_matrix)-1)
        all_so_tran[id_matrix] += 1
        # print('check', file_temp)
        file_temp = [all_matrix[id_matrix][0], all_matrix[id_matrix][1], all_matrix[id_matrix][2], id_matrix]

    action = neural_network(state, file_temp, list_action)

    if action not in list_action:
        raise Exception('action sai', list_action, action)
    check = getReward(state)
    if check != -1:
        if check == 1:
            if type(file_per[0]) == int:
                file_per = [file_temp[:3]]
            else:
                file_per.append(file_temp[:3])
            all_matrix_score[file_temp[3]] += getAgentSize() - 1
        else:
            all_matrix_score[file_temp[3]] -= 1
    return action, file_temp, file_per

def neural_network(state, file_temp, list_action):
    norm_state = state*state
    norm_action = np.zeros(AMOUNT_ACTION)
    norm_action[list_action] = 1
    norm_action = norm_action.reshape(1, AMOUNT_ACTION)
    norm_state = (norm_state/np.linalg.norm(norm_state, 1)).reshape(1,LEN_STATE)
    norm_state = norm_state/np.log(norm_state)
    hs1 = norm_state @ file_temp[0]
    # print('hs1', hs1.shape)
    hs1 = sigmoid(hs1)
    # print(hs1.shape, file_temp[1].shape)
    hs2 = hs1 @ file_temp[1]
    hs2 = sigmoid(hs2)
    # print(hs2.shape, file_temp[2].shape)
    hs3 = hs2 @ file_temp[2]
    # print('hs3', hs3.shape)
    hs3 = np.tanh(hs3)

    result_val_action = hs3*norm_action
    action_max = np.argmax(result_val_action)
    # if action_max not in list_action:
    #     print(result_val_action)
    #     print(file_temp)
    #     raise Exception('action sai', list_action, action_max)
    return action_max

#bộ hệ số 2

def test(state, file_temp, file_per):
    player = 'Hieu_200922' #Tên folder
    path_save_player = f'Agent/{player}/Data/{game_name}_{time_run_game}/'
    list_action = getValidActions(state)
    
    if len(file_temp) < 2:
        # print('đén đấy')
        with open(f'{path_save_player}best_matrix.npy', 'rb') as outfile:
            best_matrix = np.load(outfile, allow_pickle= True)
        # print('qua')
        file_temp = best_matrix
    action = neural_network(state, file_temp, list_action)
    if action not in list_action:
        raise Exception('action sai', list_action, action)
    return action, file_temp, file_per

def get_random_matrix():
    global all_matrix, all_matrix_score, all_so_tran, all_ratio
    list_player = [Hieu_player1]*(getAgentSize() - 3) + [random_player]*3        #đấu ngẫu nhiên, lấy các bot thắng random
    result1, data_save1 = normal_main(list_player,20, [0])
    # print(result1)
    if type(data_save1[0]) != int:
        all_matrix = all_matrix + data_save1
        all_matrix_score = np.append(all_matrix_score, np.zeros(len(data_save1)))
        all_so_tran = np.append(all_so_tran, np.zeros(len(data_save1)))
        all_ratio = np.append(all_ratio, np.zeros(len(data_save1)))
    # print(len(all_matrix))

def train(x):
    global all_matrix, all_matrix_score, all_so_tran, all_ratio, ratio_win
    while len(all_matrix) < 10:
        #B1: lấy đủ matrix để lọc
        get_random_matrix()
        # print('chạy lấy matrix', len(all_matrix))

    # all_matrix_score = np.zeros(len(all_matrix))
    # all_so_tran = np.zeros(len(all_matrix))
    # all_ratio = np.zeros(len(all_matrix))
    list_player = [Hieu_player_2]*(getAgentSize()-1) + [random_player]
    result, data_save = normal_main(list_player,len(all_matrix)**2, [0])
    all_so_tran = np.where(all_so_tran == 0, 1000, all_so_tran)     #loại các ma trận k được thi đấu
    all_ratio = all_matrix_score/all_so_tran
    ratio_win = (all_matrix_score+all_so_tran)/(all_so_tran*getAgentSize())
    id_best_matrix = np.argmax(all_ratio*ratio_win)
    best_matrix = all_matrix[id_best_matrix]
    np.save(f'{path_save_player}best_matrix.npy', best_matrix)
    # with open(f'{path_save_player}top_matrix.npy', 'wb') as outfile:
    #             np.save(outfile, all_matrix)
    # print('DONE PRE')
    while True:
        # print('loop DEA')
        # list_player = [random_player]*(getAgentSize()-1) + [test]
        # result, data_save = normal_main(list_player,1000, [0])
        # # print(result)
        DEA(best_matrix)

def Hieu_player_DEA(state, file_temp, file_per):
    global trial_matrix
    list_action = getValidActions(state)
    if len(file_temp) < 2:
        # id_matrix = np.random.randint(0, len(all_matrix))
        # all_so_tran[id_matrix] += 1
        file_temp = trial_matrix

    action = neural_network(state, file_temp, list_action)

    if action not in list_action:
        raise Exception('action sai')

    return action, file_temp, file_per

def DEA(best_matrix):
    global all_new_matrix, all_matrix, all_matrix_score, all_so_tran, all_ratio, ratio_win
    # print(len(all_matrix))
    get_random_matrix()
    # print(len(all_matrix))
    all_new_matrix = all_matrix.copy()
    all_index = np.arange(len(all_matrix))
    for i in range(len(all_matrix)):
        cross_matrix(all_index, best_matrix)

    if len(all_new_matrix) > len(all_matrix):
        all_matrix = all_new_matrix.copy()
        all_matrix_score = np.zeros(len(all_matrix))
        all_so_tran = np.zeros(len(all_matrix))
        all_ratio = np.zeros(len(all_matrix))
        list_player = [Hieu_player_2]*(getAgentSize()-2) + [random_player, random_player]
        # print('số trận', len(all_matrix)**2)
        result, data_save = normal_main(list_player,len(all_matrix)**2, [0])
        all_so_tran = np.where(all_so_tran == 0, -1000, all_so_tran)     #loại các ma trận k được thi đấu
        all_ratio = all_matrix_score/all_so_tran
        ratio_win = (all_matrix_score+all_so_tran)/(all_so_tran*getAgentSize())
        id_best_matrix = np.argmax(all_ratio*ratio_win)
        if id_best_matrix != 0:
            best_matrix_new = all_matrix[id_best_matrix]
            best_matrix = choose_better_matrix(best_matrix, best_matrix_new)
            with open(f'{path_save_player}best_matrix.npy', 'wb') as outfile:
                        np.save(outfile, best_matrix)
        len_top = 8
        top_top = heapq.nlargest(len_top,all_ratio)         #lấy top len_top giá trị lớn nhất
        all_top_index = np.array([-1])
        for top in top_top:
            all_top_index = np.append(all_top_index, np.where(all_ratio==top)[0])
        list_index_top = np.array(np.unique(all_top_index)[1:len_top+1]).astype(np.int64)
        all_matrix_old = all_matrix.copy()
        all_matrix = []
        for id_matrix in list_index_top:
            all_matrix.append(all_matrix_old[id_matrix])

def cross_matrix(all_index, best_matrix):
    global all_new_matrix, all_matrix, all_matrix_score, all_so_tran, all_ratio, ratio_win
    ratio_p = np.max(ratio_win)
    F = 1/len(all_matrix)
    random_2_matrix = np.random.choice(all_index, 2, replace=False)
    matrix1 = all_matrix[random_2_matrix[0]]
    matrix2 = all_matrix[random_2_matrix[1]]
    for loop in range(5):
        deno_matrix = best_matrix.copy()
        # print(len(best_matrix))
        for i in range(len(deno_matrix)):
            deno_matrix[i] = deno_matrix[i] + F*(matrix1[i] - matrix2[i])
            if i == 0:
                deno_matrix[i] = np.where(deno_matrix[i] > 2, 2, deno_matrix[i])
            else:
                deno_matrix[i] = np.where(deno_matrix[i] > 1, 1, deno_matrix[i])

            deno_matrix[i] = np.where(deno_matrix[i] < -1, -1, deno_matrix[i])
        for i in range(len(deno_matrix)):
            # print(deno_matrix[i].shape)
            random_p = np.random.randn(deno_matrix[i].shape[0], deno_matrix[i].shape[1])
            trial_matrix[i] = np.where(random_p > ratio_p, deno_matrix[i], best_matrix[i])

        list_player = [random_player]*(getAgentSize()-3) + [Hieu_player_2, Hieu_player_2, Hieu_player_DEA] 
        result_cr, data_save_cr = normal_main(list_player,len(all_matrix), [0]) 
        if np.argmax(result_cr) == getAgentSize()-1:
            all_new_matrix.append(trial_matrix)

def choose_better_matrix(best_matrix, best_matrix_new):
    global trial_matrix
    trial_matrix = best_matrix_new
    result = np.zeros(getAgentSize())
    list_player = [Hieu_player_DEA, test, Hieu_player_2] + [random_player]*(getAgentSize()-3)
    result_cr, data_save_cr = normal_main(list_player, 1000, [0]) 
    result[0] += result_cr[0]
    result[1] += result_cr[1]

    # print('new_bot_old_bot', result_cr)

    list_player = [Hieu_player_DEA, random_player, random_player]  + [random_player]*(getAgentSize()-3)
    result_cr, data_save_cr = normal_main(list_player, 1000, [0]) 
    result[0] += result_cr[0]
    # print('new_bot_random', result_cr)

    list_player = [random_player, test, random_player]  + [random_player]*(getAgentSize()-3) 
    result_cr, data_save_cr = normal_main(list_player, 1000, [0]) 
    result[1] += result_cr[1]
    # print('old_bot_random', result_cr)
    # print('choose_matrix', result)
    if result[1] < result[0]*0.975:
        best_matrix = best_matrix_new

    return best_matrix


