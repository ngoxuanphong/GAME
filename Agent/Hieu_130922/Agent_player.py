import math
import numpy as np
import heapq
from scipy.stats.mstats import gmean, hmean
import json
import random

random.seed(100)
import os
import sys
from setup import game_name,time_run_game
sys.path.append(os.path.abspath(f"base/{game_name}"))
from env import *

player = 'Hieu_130922'  #Tên folder của người chơi
path_data = f'Agent/{player}/Data'
if not os.path.exists(path_data):
    # print("folder")
    os.mkdir(path_data)
path_save_player = f'Agent/{player}/Data/{game_name}_{time_run_game}/'
if not os.path.exists(path_save_player):
    os.mkdir(path_save_player)

def silu(x, theda = 1.0):
    return x * sigmoid(theda *x)

def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))
    return sig

def Hieu_player1(state, file_temp, file_per):
    list_action = getValidActions(state)
    list_action = np.where(list_action == 1)[0]
    if len(file_temp) < 2:
        file_temp = [0, 0, 0]
        raw1 = len(state)
        col1 = min(getActionSize(), len(state))
        raw2 = col1
        col2 = len(state)//getAgentSize()
        raw3 = col2
        col3 = getActionSize()
        file_temp[0] = np.random.rand(raw1, col1)*3-1
        file_temp[1] = np.random.rand(raw2, col2)*2-1
        file_temp[2] = np.random.rand(raw3, col3)
    action = neural_network(state, file_temp, list_action)
    if action not in list_action:
        raise Exception('action sai')
    if getReward(state) == 1:
        if type(file_per[0]) == int:
            file_per = [file_temp]
        else:
            # if len(file_per) % 2000 == 0:
            #     print(len(file_per))
            file_per.append(file_temp)
    return action, file_temp, file_per

def test(state, file_temp, file_per):
    player = 'Hieu_130922' #Tên folder
    path_save_player = f'Agent/{player}/Data/{game_name}_{time_run_game}/'
    
    list_action = getValidActions(state)
    list_action = np.where(list_action == 1)[0]
    if len(file_temp) < 2:
        with open(f'{path_save_player}CK_Matran.npy', 'rb') as outfile:
            best_matrix = np.load(outfile, allow_pickle= True)
        file_temp = best_matrix
    action = neural_network(state, file_temp, list_action)
    # if action not in list_action:
    #     raise Exception('action sai')
    return action, file_temp, file_per

all_matrix  = []
all_matrix_score = []
all_so_tran = []
all_ratio = []

def Hieu_player_2(state, file_temp, file_per):
    global all_matrix, all_matrix_score, all_so_tran
    list_action = getValidActions(state)
    list_action = np.where(list_action == 1)[0]
    if len(file_temp) < 2:
        id_matrix = np.random.randint(0, len(all_matrix))
        all_so_tran[id_matrix] += 1
        file_temp = [all_matrix[id_matrix][0], all_matrix[id_matrix][1], all_matrix[id_matrix][2], id_matrix]

    action = neural_network(state, file_temp, list_action)

    if action not in list_action:
        raise Exception('action sai')
    check = getReward(state)
    if check != -1:
        if check == 1:
            if type(file_per[0]) == int:
                file_per = [file_temp[:3]]
            else:
                file_per.append(file_temp[:3])
            all_matrix_score[file_temp[3]] += 2
        else:
            all_matrix_score[file_temp[3]] -= 1
    return action, file_temp, file_per

def neural_network(state, file_temp, list_action):
    # print('check1',state)
    norm_state = (state/np.linalg.norm(state, 1)).reshape(1,len(state))
    # print('check2', norm_state)
    # norm_state = 1 / (1 + np.exp(-norm_state))          #dạng sigmoid
    # norm_state = np.arctan(norm_state)                     #dạng sin
    norm_state = np.tanh(norm_state)                    #dạng tanh
    # print('check3', norm_state)
    norm_action = np.zeros(getActionSize())
    norm_action[list_action] = 1
    norm_action = norm_action.reshape(1, getActionSize())
    # print(matrix1.shape)
    matrixRL1 = norm_state@file_temp[0]
    # print(matrixRL1.shape)
    # matrixRL1 = silu(matrixRL1, theda = 1.0)
    matrixRL1 = matrixRL1*(matrixRL1 > 0)           #activation = relu
    # print(matrixRL1.shape, file_temp[1].shape)
    matrixRL2 = matrixRL1@file_temp[1]
    matrixRL2 = 1 / (1 + np.exp(-matrixRL2))            #activation = sigmoid
    # matrixRL2 = matrixRL2*(matrixRL2 > 0)           #activation = relu
    matrixRL3 = matrixRL2@file_temp[2]
    matrixRL3 = np.tanh(matrixRL3)              #activation = tanh
    # matrixRL3 = np.sin(matrixRL3)               #activation = sin
    result_val_action = matrixRL3*norm_action
    # print(matrixRL3.shape)
    action_max = np.argmax(result_val_action)
    # print(matrixRL3)
    return action_max

def random_player(player_state, file_temp, file_per):
    list_action = getValidActions(player_state)
    list_action = np.where(list_action == 1)[0]
    action = int(np.random.choice(list_action))
    return action, file_temp, file_per

#bộ hệ số 2
def filter_matrix(save_matrix):
    global all_matrix, all_matrix_score, all_so_tran, a1, a2
    all_matrix = save_matrix.copy()
    all_matrix_score = np.zeros(len(all_matrix))
    all_so_tran = np.zeros(len(all_matrix))
    all_ratio = np.zeros(len(all_matrix))
    #VÒng loại lượt 1
    random1 = 1500
    list_player = [Hieu_player_2]*(getAgentSize()-1) + [random_player]
    result2, data_save2 = normal_main(list_player,random1, [0])
    all_so_tran = np.where(all_so_tran == 0, 1000, all_so_tran)     #loại các ma trận k được thi đấu
    all_ratio = all_matrix_score/all_so_tran
    
    len_top1 = len(all_matrix)//10
    top_top = heapq.nlargest(len_top1,all_ratio)         #lấy top len_top1 giá trị lớn nhất
    all_top_index = np.array([-1])
    for top in top_top:
        all_top_index = np.append(all_top_index, np.where(all_ratio==top)[0])
    list_index_top = np.array(np.unique(all_top_index)[1:len_top1]).astype(np.int64)
    all_matrix_old = all_matrix.copy()
    
    
    #Vòng loại lượt 2
    all_matrix = []
    for id_matrix in list_index_top:
        all_matrix.append(all_matrix_old[id_matrix])
    all_matrix_score = np.zeros(len(all_matrix))
    all_so_tran = np.zeros(len(all_matrix))
    all_ratio = np.zeros(len(all_matrix))
    random2 = 450
    list_player = [Hieu_player_2]*(getAgentSize()-1) + [random_player]
    result2, data_save2 = normal_main(list_player,random2, [0])
    all_so_tran = np.where(all_so_tran == 0, 1000, all_so_tran)     #loại các ma trận k được thi đấu
    all_ratio = all_matrix_score/all_so_tran

    len_top2 = 5
    top_top = heapq.nlargest(len_top2,all_ratio)         #lấy top len_top2 giá trị lớn nhất
    all_top_index = np.array([-1])
    for top in top_top:
        all_top_index = np.append(all_top_index, np.where(all_ratio==top)[0])
    list_index_top = np.array(np.unique(all_top_index)[1:len_top2]).astype(np.int64)
    all_matrix_old = all_matrix.copy()

    #Vòng cuối cùng
    all_matrix = []
    for id_matrix in list_index_top:
        all_matrix.append(all_matrix_old[id_matrix])
    all_matrix_score = np.zeros(len(all_matrix))
    all_so_tran = np.zeros(len(all_matrix))
    all_ratio = np.zeros(len(all_matrix))
    random3 = 50
    list_player = [Hieu_player_2]*(getAgentSize()-1) + [random_player]
    result2, data_save2 = normal_main(list_player,random3, [0])
    all_so_tran = np.where(all_so_tran == 0, 1000, all_so_tran)     #loại các ma trận k được thi đấu
    all_ratio = all_matrix_score/all_so_tran
    id_best_matrix = np.argmax(all_ratio)
    best_matrix = all_matrix[id_best_matrix]
    with open(f'{path_save_player}CK_Matran.npy', 'wb') as outfile:
                np.save(outfile, best_matrix)
    return best_matrix


def train(x):
    count_loop = 0
    n_game = 10000*x
    global all_matrix, all_matrix_score, all_so_tran
    random1 = 2000
    n_game -= random1
    list_player = [Hieu_player1]*(getAgentSize() - 2) + [random_player]*2        #đấu ngẫu nhiên, lấy các bot thắng random
    result1, data_save1 = normal_main(list_player,random1, [0])
    # print(f'Done đấu random bắt đầu________thu được {len(data_save1)}_______________kết quả_______{result1}' )
    best_matrix = filter_matrix(data_save1)
    n_game -= 2000
    # print(f'Done đấu tính điểm________')
    temp_data_save_loop = []
    while n_game > 0:
        list_player = [Hieu_player1]*(getAgentSize() - 2) + [random_player, test]
        result_loop, data_save_loop = normal_main(list_player,1000, [0])
        n_game -= 1000
        # print(result_loop,'ket qua ra soat loop', len(data_save_loop))
        data_save_loop.append(best_matrix)
        if len(data_save_loop) > 100 and count_loop == 0:
            best_matrix = filter_matrix(data_save_loop)
            # print(f'Done đấu tính điểm________')
            n_game -= 2000
        else:
            if len(temp_data_save_loop) == 0 and count_loop == 0:
                count_loop = 3
            elif len(temp_data_save_loop) > 0 and count_loop == 0:
                count_loop = 4
            temp_data_save_loop = temp_data_save_loop + data_save_loop
            
            count_loop -= 1
            if len(temp_data_save_loop) > 100 and count_loop == 0:
                data_save_loop = temp_data_save_loop.copy()
                temp_data_save_loop = []
                # print(len(data_save_loop))
                best_matrix = filter_matrix(data_save_loop)
                # print(f'Done đấu tính điểm________')
                n_game -= 2000
    
    return best_matrix



















