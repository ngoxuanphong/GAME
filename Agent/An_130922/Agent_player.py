import numpy as np
import os
import sys

from setup import game_name,time_run_game
sys.path.append(os.path.abspath(f"base/{game_name}"))
from env import *
player = 'An_130922'
path_data = f'Agent/{player}/Data'
if not os.path.exists(path_data):
    os.mkdir(path_data)

# player = 'An_130922'
path_save_player = f'Agent/{player}/Data/{game_name}_{time_run_game}/'
if not os.path.exists(path_save_player):
    os.mkdir(path_save_player)

def Identity(x):
    '''
    (-inf, inf)
    '''
    return x

def BinaryStep(x):
    '''
    {0, 1}
    '''
    x[x>=0] = 1.0
    x[x<0] = 0.0
    return x

def Sigmoid(x):
    '''
    (0, 1)
    '''
    return 1.0 / (1.0 + np.e**(-x))

def NegativePositiveStep(x):
    '''
    {-1, 1}
    '''
    x[x>=0] = 1.0
    x[x<0] = -1.0
    return x

def Tanh(x):
    '''
    (-1, 1)
    '''
    return (np.e**(x) - np.e**(-x)) / (np.e**(x) + np.e**(-x))

def ReLU(x):
    '''
    [0, inf)
    '''
    return x * (x>0)

def LeakyReLU(x):
    '''
    (-inf, inf)
    '''
    x[x<0] *= 0.01
    return x

def PReLU(x, a=0.5):
    '''
    (-inf, inf)
    '''
    x[x<0] *= 0.5
    return x

def Gaussian(x):
    '''
    (0, 1]
    '''
    return np.e**(-x**2)

list_activation_function = [Identity, BinaryStep, Sigmoid, NegativePositiveStep, Tanh, ReLU, LeakyReLU, PReLU, Gaussian]

def rdb(p_state, temp_file, per_file):
    list_action = getValidActions(p_state)
    list_action = np.where(list_action == 1)[0]
    action = np.random.choice(list_action)
    return action, temp_file, per_file

def neural_network(p_state, data, list_action):
    '''
    temp_file: Một list độ dài chẵn: [matrix, activation_fun, ...]
    OUTPUT: action_max
    '''

    res_mat = p_state.copy()
    for i in range(len(data)):
        if i % 2 == 0:
            res_mat = res_mat @ data[i]
            max_x = np.max(np.abs(res_mat))
            max_x_1 = max_x/25
            res_mat = res_mat / max_x_1
        else:
            res_mat = list_activation_function[data[i]](res_mat)
    
    res_arr = res_mat[list_action]
    arr_max = np.where(res_arr == np.max(res_arr))[0]
    action_max_idx = np.random.choice(arr_max)
    return list_action[action_max_idx]

def generate_data(p_state, temp_file, per_file):
    if len(temp_file) < 2:
        temp_file = []
        len_state = len(p_state)
        total_action = getActionSize()

        k = total_action * len_state
        if k < 1000:
            number_layer = 7
        elif 1000 <= k and k < 8000:
            number_layer = 5
        elif 8000 <= k and k < 24000:
            number_layer = 5
        elif 24000 <= k:
            number_layer = 3
        
        nRow = len_state
        nCol = -1
        for i in range(number_layer-2):
            nCol = np.round((len_state/total_action)**((number_layer-2-i)/(number_layer-1))*total_action).astype(int)
            matrix = np.random.uniform(-1.0,1.0,(nRow, nCol))
            nRow = nCol
            temp_file.append(matrix)
            temp_file.append(np.random.randint(0,9))
        
        nCol = total_action
        matrix = np.random.uniform(-1.0,1.0,(nRow, nCol))
        temp_file.append(matrix)
        temp_file.append(np.random.randint(0,9))

    list_action = getValidActions(p_state)
    list_action = np.where(list_action == 1)[0]
    action = neural_network(p_state, temp_file, list_action)

    if getReward(p_state) == 1:
        per_file.append(temp_file)

    return action, temp_file, per_file

def scoring(p_state, temp_file, per_file):
    '''
    per_file: [
        tất cả data về neural_network,
        array thể hiện số lần thi đấu thắng,
        array thể hiên số lần tham gia thi đấu
    ]
    '''
    if len(temp_file) < 2:
        k = np.random.randint(0, len(per_file[0]))
        temp_file = [per_file[0][k], k]
        per_file[2][k] += 1
    
    list_action = getValidActions(p_state)
    list_action = np.where(list_action == 1)[0]
    action = neural_network(p_state, temp_file[0], list_action)

    if getReward(p_state) == 1:
        per_file[1][temp_file[1]] += 1
    
    return action, temp_file, per_file

def test(p_state, temp_file, per_file):
    if len(temp_file) < 2:
        player = 'An_130922'
        path_save_player = f'Agent/{player}/Data/{game_name}_{time_run_game}/'
        temp_file = np.load(f'{path_save_player}/Ahih1st.npy', allow_pickle=True)
    
    list_action = getValidActions(p_state)
    list_action = np.where(list_action == 1)[0]
    action = neural_network(p_state, temp_file, list_action)
    return action, temp_file, per_file

def train(num_cycle=1):
    for _ in range(num_cycle):
        try:
            np.load(f'{path_save_player}/Ahih1st.npy', allow_pickle=True)
            cur_best_bot = test
            per_file = [np.load(f'{path_save_player}/Ahih1st.npy', allow_pickle=True)]
        except:
            cur_best_bot = rdb
            per_file = []
    
        # Generate
        list_player = [generate_data] + [cur_best_bot] + [rdb] * (getAgentSize() - 2)
        kq, per_file = normal_main(list_player, 1000, per_file)
        list_player = [generate_data] + [rdb] * (getAgentSize() - 1)
        kq, per_file = normal_main(list_player, 1000, per_file)

        np.save(f'{path_save_player}/Ahihies.npy', per_file)

        # Scoring
        list_player = [scoring] * (getAgentSize()-1) + [rdb]
        num_data = len(per_file)
        per_file = [per_file, np.ones(num_data*100), np.full(num_data*100, getAgentSize()*100)]
        kq, per_file = normal_main(list_player, 3000, per_file)
        score = per_file[1] / per_file[2]
        arr_max = np.where(score == max(score))[0]
        cur_best_data = per_file[0][np.random.choice(arr_max)]
        np.save(f'{path_save_player}/Ahih1st.npy', cur_best_data)

        # Tối ưu từng phần
        for _k_ in range(10):
            for idx_can_toi_uu in range(len(cur_best_data)):
                per_file = [cur_best_data]
                if idx_can_toi_uu % 2 == 0:
                    num_match = 3000
                    matrix_shape = cur_best_data[idx_can_toi_uu].shape
                    for _ in range(300):
                        new_matrix = np.random.uniform(-1.0,1.0,matrix_shape)
                        new_data = cur_best_data.copy()
                        new_data[idx_can_toi_uu] = new_matrix
                        per_file.append(new_data)
                else:
                    num_match = 1000
                    old_value = cur_best_data[idx_can_toi_uu]
                    for i in range(9):
                        if i != old_value:
                            new_value = i
                            new_data = cur_best_data.copy()
                            new_data[idx_can_toi_uu] = new_value
                            per_file.append(new_data)
                
                num_data = len(per_file)
                per_file = [per_file, np.ones(num_data*100), np.full(num_data*100, getAgentSize()*100)]
                list_player = [scoring] * (getAgentSize()-1) + [rdb]
                kq, per_file = normal_main(list_player, num_match, per_file)
                score = per_file[1] / per_file[2]
                arr_max = np.where(score == max(score))[0]
                cur_best_data = per_file[0][np.random.choice(arr_max)]
                np.save(f'{path_save_player}/Ahih1st.npy', cur_best_data)
    
    return cur_best_data

# train(1)

# print('Test')
# list_player = [test] + [rdb] * (getAgentSize() - 1)
# print(normal_main(list_player, 1000, []))