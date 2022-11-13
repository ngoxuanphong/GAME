import numpy as np
import heapq
from scipy.stats.mstats import gmean, hmean
import itertools

import os
import sys
from setup import game_name,time_run_game
sys.path.append(os.path.abspath(f"base/{game_name}"))
from env import *

player = 'Hieu_270922'  #Tên folder của người chơi
path_data = f'Agent/{player}/Data'
if not os.path.exists(path_data):
    # print("folder")
    os.mkdir(path_data)
path_save_player = f'Agent/{player}/Data/{game_name}_{time_run_game}/'
if not os.path.exists(path_save_player):
    os.mkdir(path_save_player)


def random_player(player_state, file_temp, file_per):
    list_action = getValidActions(player_state)
    list_action = np.where(list_action == 1)[0]
    action = int(np.random.choice(list_action))
    return action, file_temp, file_per

def agent(state,file_temp,file_per):
    list_action = getValidActions(state)
    list_action = np.where(list_action == 1)[0]
    action = np.random.choice(list_action)
    file_per = (len(state),getActionSize())
    return action,file_temp,file_per

LEN_STATE,AMOUNT_ACTION = normal_main([agent]*getAgentSize(), 1, [0])[1]
AMOUNT_PLAYER = getAgentSize()
MINMIN = min(AMOUNT_ACTION, LEN_STATE)

def relu(X):
    '''It returns zero if the input is less than zero otherwise it returns the given input.'''
    return np.maximum(0,X)

def softmax(X):
    ''' Compute softmax values for each sets of scores in x. '''
    expo = np.exp(X)
    return expo/np.sum(expo)

def sigmoid(X):
    ''' It returns 1/(1+exp(-x)). where the values lies between zero and one '''
    return 1/(1+np.exp(-X))

def tanh(X):
    ''' It returns the value (1-exp(-2x))/(1+exp(-2x)) and the value returned will be lies in between -1 to 1.'''
    return np.tanh(X)

def leaky_relu(x, theta = 0.05):
    '''It returns input * theta if the input is less than zero otherwise it returns the given input.'''
    return np.where(x > 0, x, x * theta)

def silu(x, theta = 0.5):
    return x * sigmoid(theta *x)

def Softplus_grad(x): 
    return np.divide(1.,1.+np.exp(-x))

def Softplus(x): 
    return np.log(1 + np.exp(-np.abs(x))) + np.maximum(x,0) 

def Hieu_player1(state, file_temp, file_per):
    list_action = getValidActions(state)
    list_action = np.where(list_action == 1)[0]
    if len(file_temp) < 2:
        # file_temp = [0, 0, 0]
        raw1 = len(state)
        col1 = MINMIN
        raw2 = col1
        col2 = len(state)//AMOUNT_PLAYER
        raw3 = col2
        col3 = AMOUNT_ACTION
        file_temp = [np.random.uniform(-1, 1, [raw1, col1]),
                    np.random.uniform(-1, 1, [raw2, col2]),
                    np.random.uniform(-1, 1, [raw3, col3])]
    action = neural_network(state, file_temp, list_action)
    
    if getReward(state) == 1:
        if type(file_per[0]) == int:
            file_per = [file_temp]
        else:
            # if len(file_per) % 2000 == 0:
            #     print(len(file_per))
            file_per.append(file_temp)
    return action, file_temp, file_per


def neural_network(state, file_temp, list_action):
    norm_state = state.copy()
    norm_state = (norm_state/np.linalg.norm(norm_state, 1)).reshape(1,LEN_STATE)
    norm_state = softmax(norm_state)
    norm_action = np.zeros(AMOUNT_ACTION)
    norm_action[list_action] = 1
    norm_action = norm_action.reshape(1, AMOUNT_ACTION)

    matrixRL1 = norm_state@file_temp[0]
    matrixRL1 = sigmoid(matrixRL1)          

    matrixRL2 = matrixRL1@file_temp[1]
    matrixRL2 = tanh(matrixRL2)         

    matrixRL3 = matrixRL2@file_temp[2]
    matrixRL3 = softmax(matrixRL3)   

    result_val_action = matrixRL3*norm_action
    action_max = np.argmax(result_val_action)
    # if action_max not in list_action:
    #     print(action_max, list_action, result_val_action)
    #     raise Exception('action sai')
    return action_max




def test(state, file_temp, file_per):
    player = 'Hieu_270922' #Tên folder
    path_save_player = f'Agent/{player}/Data/{game_name}_{time_run_game}/'
    list_action = getValidActions(state)
    list_action = np.where(list_action == 1)[0]
    if len(file_temp) < 2:
        with open(f'{path_save_player}best_matrix.npy', 'rb') as outfile:
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
all_id_matrix = []
trial_matrix = [0, 0, 0]
ratio_win = []

def Hieu0(state, file_temp, file_per):
    global all_matrix, all_matrix_score, all_so_tran, all_ratio, ratio_win, trial_matrix, all_id_matrix
    list_action = getValidActions(state)
    list_action = np.where(list_action == 1)[0]
    if len(file_temp) < 2:
        id_matrix = all_id_matrix[0]
        all_so_tran[id_matrix] += 1
        file_temp = [all_matrix[id_matrix][0], all_matrix[id_matrix][1], all_matrix[id_matrix][2], id_matrix]

    action = neural_network(state, file_temp, list_action)

    # if action not in list_action:
    #     raise Exception('action sai')
    check = getReward(state)
    if check != -1:
        if check == 1:
            if type(file_per[0]) == int:
                file_per = [file_temp[:3]]
            else:
                file_per.append(file_temp[:3])
            all_matrix_score[file_temp[3]] += AMOUNT_PLAYER-1
        else:
            all_matrix_score[file_temp[3]] -= 1
    return action, file_temp, file_per

def Hieu1(state, file_temp, file_per):
    global all_matrix, all_matrix_score, all_so_tran, all_ratio, ratio_win, trial_matrix, all_id_matrix
    list_action = getValidActions(state)
    list_action = np.where(list_action == 1)[0]
    if len(file_temp) < 2:
        id_matrix = all_id_matrix[1]
        all_so_tran[id_matrix] += 1
        file_temp = [all_matrix[id_matrix][0], all_matrix[id_matrix][1], all_matrix[id_matrix][2], id_matrix]

    action = neural_network(state, file_temp, list_action)

    # if action not in list_action:
    #     raise Exception('action sai')
    check = getReward(state)
    if check != -1:
        if check == 1:
            if type(file_per[0]) == int:
                file_per = [file_temp[:3]]
            else:
                file_per.append(file_temp[:3])
            all_matrix_score[file_temp[3]] += AMOUNT_PLAYER-1
        else:
            all_matrix_score[file_temp[3]] -= 1
    return action, file_temp, file_per

def Hieu2(state, file_temp, file_per):
    global all_matrix, all_matrix_score, all_so_tran, all_ratio, ratio_win, trial_matrix, all_id_matrix
    list_action = getValidActions(state)
    list_action = np.where(list_action == 1)[0]
    if len(file_temp) < 2:
        id_matrix = all_id_matrix[2]
        all_so_tran[id_matrix] += 1
        file_temp = [all_matrix[id_matrix][0], all_matrix[id_matrix][1], all_matrix[id_matrix][2], id_matrix]

    action = neural_network(state, file_temp, list_action)

    # if action not in list_action:
    #     raise Exception('action sai')
    check = getReward(state)
    if check != -1:
        if check == 1:
            if type(file_per[0]) == int:
                file_per = [file_temp[:3]]
            else:
                file_per.append(file_temp[:3])
            all_matrix_score[file_temp[3]] += AMOUNT_PLAYER-1
        else:
            all_matrix_score[file_temp[3]] -= 1
    return action, file_temp, file_per

def Hieu3(state, file_temp, file_per):
    global all_matrix, all_matrix_score, all_so_tran, all_ratio, ratio_win, trial_matrix, all_id_matrix
    list_action = getValidActions(state)
    list_action = np.where(list_action == 1)[0]
    if len(file_temp) < 2:
        id_matrix = all_id_matrix[3]
        all_so_tran[id_matrix] += 1
        file_temp = [all_matrix[id_matrix][0], all_matrix[id_matrix][1], all_matrix[id_matrix][2], id_matrix]

    action = neural_network(state, file_temp, list_action)

    # if action not in list_action:
    #     raise Exception('action sai')
    check = getReward(state)
    if check != -1:
        if check == 1:
            if type(file_per[0]) == int:
                file_per = [file_temp[:3]]
            else:
                file_per.append(file_temp[:3])
            all_matrix_score[file_temp[3]] += AMOUNT_PLAYER-1
        else:
            all_matrix_score[file_temp[3]] -= 1
    return action, file_temp, file_per

def Hieu4(state, file_temp, file_per):
    global all_matrix, all_matrix_score, all_so_tran, all_ratio, ratio_win, trial_matrix, all_id_matrix
    list_action = getValidActions(state)
    list_action = np.where(list_action == 1)[0]
    if len(file_temp) < 2:
        id_matrix = all_id_matrix[4]
        all_so_tran[id_matrix] += 1
        file_temp = [all_matrix[id_matrix][0], all_matrix[id_matrix][1], all_matrix[id_matrix][2], id_matrix]

    action = neural_network(state, file_temp, list_action)

    # if action not in list_action:
    #     raise Exception('action sai')
    check = getReward(state)
    if check != -1:
        if check == 1:
            if type(file_per[0]) == int:
                file_per = [file_temp[:3]]
            else:
                file_per.append(file_temp[:3])
            all_matrix_score[file_temp[3]] += AMOUNT_PLAYER-1
        else:
            all_matrix_score[file_temp[3]] -= 1
    return action, file_temp, file_per

def Hieu_trial(state, file_temp, file_per):
    global trial_matrix
    list_action = getValidActions(state)
    list_action = np.where(list_action == 1)[0]
    if len(file_temp) < 2:
        file_temp = trial_matrix

    action = neural_network(state, file_temp, list_action)
    # if action not in list_action:
    #     raise Exception('action sai')
    return action, file_temp, file_per

def filter_matrix():
    global all_matrix, all_matrix_score, all_so_tran, all_ratio, ratio_win, trial_matrix, all_id_matrix
    # all_matrix = save_matrix.copy()
    all_id_of_matrix = np.arange(len(all_matrix))
    all_set_id_matrix = np.array(list(itertools.combinations(all_id_of_matrix, AMOUNT_PLAYER)))
    all_matrix_score = np.zeros(len(all_matrix))
    all_so_tran = np.zeros(len(all_matrix))
    all_ratio = np.zeros(len(all_matrix))
    list_player = []
    if AMOUNT_PLAYER == 4:
        list_player = [Hieu0, Hieu1, Hieu2, random_player]
    else:
        list_player = [Hieu0, Hieu1, Hieu2, Hieu3, random_player]
    for i in range(len(all_set_id_matrix)):
        all_id_matrix = all_set_id_matrix[i]
        result2, data_save2 = normal_main(list_player,AMOUNT_PLAYER*2, [0])
    
    all_ratio = all_matrix_score/all_so_tran
    ratio_win = (all_matrix_score+all_so_tran)/(all_so_tran*AMOUNT_PLAYER)

    len_top1 = AMOUNT_PLAYER
    top_top = heapq.nlargest(len_top1,ratio_win)         #lấy top len_top1 giá trị lớn nhất
    all_top_index = np.array([-1])
    for top in top_top:
        all_top_index = np.append(all_top_index, np.where(ratio_win==top)[0])
    list_index_top = np.array(np.unique(all_top_index)[1:len_top1]).astype(np.int64)
    all_matrix_old = all_matrix.copy()
    all_matrix = []
    # print('top top', len_top1, list_index_top, len(all_matrix))
    for id_matrix in list_index_top:
        all_matrix.append(all_matrix_old[id_matrix])
    # print('top top', len_top1, list_index_top, len(all_matrix))
    all_id_of_matrix = np.arange(len(all_matrix))
    all_set_id_matrix = np.array(list(itertools.combinations(all_id_of_matrix, AMOUNT_PLAYER-1)))
    all_matrix_score = np.zeros(len(all_matrix))
    all_so_tran = np.zeros(len(all_matrix))
    all_ratio = np.zeros(len(all_matrix))
    list_player = []
    if AMOUNT_PLAYER == 4:
        list_player = [Hieu0, Hieu1, Hieu2, random_player]
    else:
        list_player = [Hieu0, Hieu1, Hieu2, Hieu3, random_player]
    for i in range(len(all_set_id_matrix)):
        all_id_matrix = all_set_id_matrix[i]
        result2, data_save2 = normal_main(list_player,AMOUNT_PLAYER**2, [0])
    
    all_ratio = all_matrix_score/all_so_tran
    ratio_win = (all_matrix_score+all_so_tran)/(all_so_tran*AMOUNT_PLAYER)
    best_win = np.max(ratio_win)
    
    all_id_win = np.where(ratio_win == best_win)[0]
    top1 = len(all_id_win)
    # if top1 < AMOUNT_PLAYER:
    best_matrix = all_matrix[all_id_win[0]]
    all_matrix_old = all_matrix.copy()
    all_matrix = []
    for id_matrix in all_id_win:
        all_matrix.append(all_matrix_old[id_matrix])
    # all_matrix = [best_matrix]
    all_matrix_score = np.zeros(top1)
    all_so_tran = np.zeros(top1)
    all_ratio = np.zeros(top1)
    ratio_win = np.zeros(top1)
    all_id_matrix = []
    return best_matrix
    # else:
    #     all_matrix_old = all_matrix.copy()
    #     all_matrix = []
    #     for id_matrix in all_id_win:
    #         all_matrix.append(all_matrix_old[id_matrix])
    #     # save_matrix = all_matrix.copy()
    #     all_matrix_score = np.zeros(top1)
    #     all_so_tran = np.zeros(top1)
    #     all_ratio = np.zeros(top1)
    #     ratio_win = np.zeros(top1)
    #     all_id_matrix = []
    #     print('đi nhánh đẹ quy', len(all_matrix), len(ratio_win), top1, all_id_win)
    #     return filter_matrix()

def get_random_matrix():
    global all_matrix, all_matrix_score, all_so_tran, all_ratio, ratio_win, trial_matrix, all_id_matrix
    list_player = [Hieu_player1] + [random_player]*(AMOUNT_PLAYER-1)        #đấu ngẫu nhiên, lấy các bot thắng random
    result1, data_save1 = normal_main(list_player,20, [0])
    # print(result1)
    if type(data_save1[0]) != int:
        all_matrix = all_matrix + data_save1
     

def train(x):
    global all_matrix, all_matrix_score, all_so_tran, all_ratio, ratio_win, trial_matrix, all_id_matrix
    while len(all_matrix) < 10:
        get_random_matrix()
    best_matrix = filter_matrix()
    with open(f'{path_save_player}best_matrix.npy', 'wb') as outfile:
        np.save(outfile, best_matrix)
    list_player = [random_player]*(AMOUNT_PLAYER-1) + [test]
    result1, data_save1 = normal_main(list_player,1000, [0])
    # print(result1)
    while True:
        k = len(all_matrix)
        while len(all_matrix) < 10+k:
            get_random_matrix()
        all_matrix.append(best_matrix)
        trial_matrix = filter_matrix()
        list_player = [test, Hieu_trial] + [random_player]*(AMOUNT_PLAYER-2) 
        result1, data_save1 = normal_main(list_player,500, [0])

        list_player = [test, random_player] + [random_player]*(AMOUNT_PLAYER-2) 
        result_temp, data_save1 = normal_main(list_player,500, [0])
        result1[0] += result_temp[0]

        list_player = [random_player, Hieu_trial] + [random_player]*(AMOUNT_PLAYER-2) 
        result_temp, data_save1 = normal_main(list_player,500, [0])
        result1[1] += result_temp[1]
        # print(result1)
        if result1[0] < result1[1]:
            best_matrix = trial_matrix
            with open(f'{path_save_player}best_matrix.npy', 'wb') as outfile:
                    np.save(outfile, best_matrix)
            list_player = [random_player]*(AMOUNT_PLAYER-1) + [test]
            result1, data_save1 = normal_main(list_player,1000, [0])
            # print(result1)

