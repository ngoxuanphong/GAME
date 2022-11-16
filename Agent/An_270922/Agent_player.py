import numpy as np
import os
import sys
from setup import game_name,time_run_game
sys.path.append(os.path.abspath(f"base/{game_name}"))
from env import *

player = 'An_270922'
path_data = f'Agent/{player}/Data'
if not os.path.exists(path_data):
    os.mkdir(path_data)

path_save_player = f'Agent/{player}/Data/{game_name}_{time_run_game}/'
if not os.path.exists(path_save_player):
    os.mkdir(path_save_player)

INF_AS_FLOAT = np.nan_to_num(np.inf)
LOG_INF = np.log(INF_AS_FLOAT)
HALF_LOG_INF = LOG_INF/2
SQRT_LOG_INF = np.sqrt(LOG_INF)

def Identity(x:np.ndarray):
    return x/np.abs(x).max()

def BinaryStep(x:np.ndarray):
    return np.where(x>=0, 1, 0)

def Sigmoid(x:np.ndarray):
    return 1/(1+np.e**(-np.where(np.abs(x)>LOG_INF, np.sign(x)*LOG_INF, x)))

def SignStep(x:np.ndarray):
    return np.sign(x)

def Tanh(x:np.ndarray):
    x_new = np.where(np.abs(x)>HALF_LOG_INF, np.sign(x)*HALF_LOG_INF, x)
    return (np.e**(2*x_new)-1)/(np.e**(2*x_new)+1)

def ReLU(x:np.ndarray):
    return np.where(x<0, 0, x)/np.max(x)

def LeakyReLU(x:np.ndarray):
    x_new = np.where(x<0, 0.01*x, x)
    return x_new/np.abs(x_new).max()

def PReLU(x:np.ndarray):
    x_new = np.where(x<0, 0.5*x, x)
    return x_new/np.abs(x_new).max()

def SoftPlus(x:np.ndarray):
    x_new = np.where(np.abs(x)>LOG_INF-1e-9, x, np.log(1+np.e**(x)))
    return x_new/np.max(x_new)

def Gaussian(x:np.ndarray):
    return np.e**(-np.where(np.abs(x)>SQRT_LOG_INF, np.sign(x)*SQRT_LOG_INF, x)**2)

activation_function = [Identity, BinaryStep, Sigmoid, SignStep, Tanh, ReLU, LeakyReLU, PReLU, SoftPlus, Gaussian]

def Ann_rdb(p_state, temp_file, per_file):
    list_action = getValidActions(p_state)
    list_action = np.where(list_action == 1)[0]
    action = np.random.choice(list_action)
    return action, temp_file, per_file

def Ann_neural_network(p_state:np.ndarray, data, list_action):
    res_mat = p_state.copy()
    for i in range(len(data)//3):
        res_mat = res_mat @ data[3*i] + data[3*i+1]
        res_mat = np.nan_to_num(res_mat)
        res_mat = activation_function[data[3*i+2]](res_mat)
    
    res_arr = res_mat[list_action]
    a = np.max(res_arr)
    if a >= 0:
        arr_max = np.where(res_arr >= 0.99*a)[0]
    else:
        arr_max = np.where(res_arr >= 1.01*a)[0]
    
    return list_action[np.random.choice(arr_max)]

def generate_fnn(p_state, temp_file, per_file):
    if len(temp_file) < 2:
        temp_file = []
        len_state = len(p_state)
        total_action = getActionSize()
        number_layer = 5
        nRow = len_state
        nCol = -1
        for i in range(number_layer-2):
            nCol = np.round((len_state/total_action)**((number_layer-2-i)/(number_layer-1))*total_action).astype(int)
            matrix = np.random.uniform(-1,1,(nRow, nCol))
            bias_arr = np.random.uniform(-len_state,len_state,nCol)
            nRow = nCol
            temp_file.append(matrix)
            temp_file.append(bias_arr)
            temp_file.append(np.random.randint(0,len(activation_function)))
        
        nCol = total_action
        matrix = np.random.uniform(-1.0,1.0,(nRow, nCol))
        bias_arr = np.random.uniform(-len_state,len_state,nCol)
        temp_file.append(matrix)
        temp_file.append(bias_arr)
        temp_file.append(np.random.randint(0,len(activation_function)))
    
    list_action = getValidActions(p_state)
    list_action = np.where(list_action == 1)[0]
    action = Ann_neural_network(p_state, temp_file, list_action)

    if getReward(p_state) == 1:
        per_file.append(temp_file)

    return action, temp_file, per_file

def generate_sg(p_state, temp_file, per_file):
    list_action = getValidActions(p_state)
    list_action = np.where(list_action == 1)[0]
    action = np.random.choice(list_action)
    per_file[1][action] += 1

    check_win = getReward(p_state)
    if check_win != -1:
        temp_file.pop(0)
        if check_win == 1:
            for i in temp_file:
                per_file[0][i] += 1
    
    temp_file.append(action)
    return action, temp_file, per_file

def test(p_state, temp_file, per_file):
    if len(temp_file) < 2:
        temp_file = np.load(f'{path_save_player}Ahih1st.npy', allow_pickle=True)
    
    list_action = getValidActions(p_state)
    list_action = np.where(list_action == 1)[0]
    if temp_file[1] == 0:
        action = Ann_neural_network(p_state, temp_file[0], list_action)
        return action, temp_file, per_file
    if temp_file[1] == 1:
        if len(temp_file) < 3:
            temp_file.append(temp_file[0][0]/temp_file[0][1])
        
        res_arr = temp_file[2][list_action]
        a = np.max(res_arr)
        arr_max = np.where(res_arr >= 0.99*a)[0]
        
        return list_action[np.random.choice(arr_max)], temp_file, per_file

def test_data(p_state, temp_file, per_file):
    list_action = getValidActions(p_state)
    list_action = np.where(list_action == 1)[0]
    if per_file[1] == 0:
        action = Ann_neural_network(p_state, per_file[0], list_action)
        return action, temp_file, per_file
    if per_file[1] == 1:
        if len(per_file) < 3:
            per_file.append(per_file[0][0]/per_file[0][1])

        res_arr = per_file[2][list_action]
        a = np.max(res_arr)
        arr_max = np.where(res_arr >= 0.99*a)[0]
        
        return list_action[np.random.choice(arr_max)], temp_file, per_file

def test_fnn(p_state, temp_file, per_file):
    if len(temp_file) < 2:
        temp_file = np.load(f'{path_save_player}Ahih1st_FNN.npy', allow_pickle=True)
    
    list_action = getValidActions(p_state)
    list_action = np.where(list_action == 1)[0]
    action = Ann_neural_network(p_state, temp_file, list_action)
    return action, temp_file, per_file

def test_sg(p_state, temp_file, per_file):
    if len(temp_file) < 2:
        temp_file = np.load(f'{path_save_player}Ahih1st_SG.npy', allow_pickle=True)
        temp_file = [temp_file[0]/temp_file[1], 0]
    
    list_action = getValidActions(p_state)
    list_action = np.where(list_action == 1)[0]
    res_arr = temp_file[0][list_action]
    a = np.max(res_arr)
    arr_max = np.where(res_arr >= 0.99*a)[0]

    return list_action[np.random.choice(arr_max)], temp_file, per_file

def scoring(p_state, temp_file, per_file):
    if len(temp_file) < 2:
        k = np.random.randint(0, len(per_file[0]))
        temp_file = per_file[0][k] + [k]
        per_file[2][k] += 1
    
    list_action = getValidActions(p_state)
    list_action = np.where(list_action == 1)[0]
    if getReward(p_state) == 1:
        per_file[1][temp_file[2]] += 1
    
    if temp_file[1] == 0:
        action = Ann_neural_network(p_state, temp_file[0], list_action)
        return action, temp_file, per_file
    if temp_file[1] == 1:
        if len(temp_file) < 4:
            temp_file.append(temp_file[0][0]/temp_file[0][1])
        
        res_arr = temp_file[3][list_action]
        a = np.max(res_arr)
        arr_max = np.where(res_arr >= 0.99*a)[0]
        
        return list_action[np.random.choice(arr_max)], temp_file, per_file

def train(num_cycle=1):
    try:
        np.load(f'{path_save_player}Ahih1st_SG.npy', allow_pickle=True)
    except:
        per_file = [np.zeros(getActionSize()), np.full(getActionSize(), 0.01)]
        list_player = [generate_sg] * getAgentSize()
        kq, per_file = normal_main(list_player, 10000, per_file)
        np.save(f'{path_save_player}Ahih1st_SG.npy', per_file)
    
    for _ in range(num_cycle):
        # print(_, 11111)
        per_file = np.load(f'{path_save_player}Ahih1st_SG.npy', allow_pickle=True)
        list_data = [per_file.copy()]

        list_player = [generate_sg] * getAgentSize()
        kq, per_file = normal_main(list_player, 10000, per_file)
        list_data.append(per_file.copy())

        new_data = [np.zeros(getActionSize()), np.full(getActionSize(), 0.01)]
        kq, new_data = normal_main(list_player, 30000, new_data)
        list_data.append(new_data.copy())

        temp_data = [np.zeros(getActionSize()), np.full(getActionSize(), 0.01)]
        for i in range(getAgentSize()-3):
            list_data.append(temp_data.copy())
        
        num_data = len(list_data)
        per_file = [[data, 1] for data in list_data]
        per_file = [per_file, np.zeros(num_data), np.full(num_data, 0.01)]
        list_player = [scoring] * (getAgentSize()-1) + [Ann_rdb]
        kq, per_file = normal_main(list_player, 3000, per_file)
        score_1 = per_file[1]/per_file[2]
        list_player = [test_data] + [Ann_rdb] * (getAgentSize()-1)
        score_2 = np.zeros(score_1.shape)
        for i in range(3):
            data = per_file[0][i]
            kq, temp = normal_main(list_player, 1000, data)
            score_2[i] = kq[0]/1000
        
        total_score = score_1 + score_2
        best_idx = total_score.argmax()
        best_data_sg = per_file[0][best_idx][0]
        np.save(f'{path_save_player}Ahih1st_SG.npy', best_data_sg)

        #
        # print(_, 22222)
        try:
            per_file = [np.load(f'{path_save_player}Ahih1st_FNN.npy', allow_pickle=True)]
            cur_best_bot = test_fnn
        except:
            per_file = []
            cur_best_bot = Ann_rdb
        
        #########################
        list_player = [generate_fnn] + [cur_best_bot] + [Ann_rdb] * (getAgentSize() - 2)
        kq, per_file = normal_main(list_player, 1000, per_file)

        list_player = [generate_fnn] + [test_sg] + [Ann_rdb] * (getAgentSize() - 2)
        kq, per_file = normal_main(list_player, 1000, per_file)

        list_player = [generate_fnn] + [Ann_rdb] * (getAgentSize() - 1)
        kq, per_file = normal_main(list_player, 1000, per_file)

        temp = per_file.copy()
        per_file = [[data, 0] for data in temp]
        per_file.append([best_data_sg, 1])

        #########################
        num_data = len(per_file)
        a_ = (1-12/num_data)*100
        if a_ < 0:
            a_ = 0
        per_file = [per_file, np.zeros(num_data), np.full(num_data, 0.01)]
        list_player = [scoring] * (getAgentSize()-1) + [Ann_rdb]
        kq, per_file = normal_main(list_player, 10000, per_file)
        score = per_file[1]/per_file[2]
        a = np.percentile(score, min(a_,99))
        arr_max = np.where(score > a)[0]
        
        #
        per_file_ = [per_file[0][i] for i in arr_max]
        num_data = len(per_file_)
        per_file = [per_file_, np.zeros(num_data), np.full(num_data, 0.01)]
        list_player = [scoring] * (getAgentSize()-1) + [Ann_rdb]
        kq, per_file = normal_main(list_player, 5000, per_file)
        score1 = per_file[1]/per_file[2]
        list_player = [test_data] + [Ann_rdb] * (getAgentSize()-1)
        score2 = np.zeros(score1.shape)
        for i in range(len(per_file[0])):
            data = per_file[0][i]
            kq, temp = normal_main(list_player, 1000, data)
            score2[i] = kq[0]/1000
        
        total_score = score1 + score2
        best_idx = total_score.argmax()
        cur_best_data = per_file[0][best_idx]
        np.save(f'{path_save_player}Ahih1st.npy', cur_best_data)

        temp = [per_file[0][i] for i in range(len(per_file[0])) if per_file[0][i][1] == 0]
        temp1 = np.array([total_score[i] for i in range(len(per_file[0])) if per_file[0][i][1] == 0])
        best_idx = temp1.argmax()
        cur_best_fnn_data = temp[best_idx][0]
        np.save(f'{path_save_player}Ahih1st_FNN.npy', cur_best_fnn_data)

        # print(_, 33333)
        for idx in range(len(cur_best_fnn_data)):
            per_file = [cur_best_fnn_data]
            list_player = [test_data] + [test_fnn] + [Ann_rdb] * (getAgentSize()-2)
            if idx % 3 in [0,1]:
                matrix_shape = cur_best_fnn_data[idx].shape
                for k in range(300):
                    new_matrix = np.random.uniform(-1.0,1.0,matrix_shape)
                    new_data = cur_best_fnn_data.copy()
                    new_data[idx] = new_matrix
                    data_test = [new_data, 0]
                    kq, p_ = normal_main(list_player, 100, data_test)
                    if kq[0] >= kq[1] / 2:
                        per_file.append(new_data)
            else:
                for i in range(len(activation_function)):
                    if i != cur_best_fnn_data[idx]:
                        new_data = cur_best_fnn_data.copy()
                        new_data[idx] = i
                        data_test = [new_data, 0]
                        kq, p_ = normal_main(list_player, 100, data_test)
                        if kq[0] >= kq[1] / 2:
                            per_file.append(new_data)
            
            num_data = len(per_file)
            if len(per_file) > 1:
                temp = per_file.copy()
                per_file = [[data, 0] for data in temp]

                #########################
                num_data = len(per_file)
                a_ = (1-12/num_data)*100
                if a_ < 0:
                    a_ = 0
                per_file = [per_file, np.zeros(num_data), np.full(num_data, 0.01)]
                list_player = [scoring] * (getAgentSize()-1) + [Ann_rdb]
                kq, per_file = normal_main(list_player, 10000, per_file)
                score = per_file[1]/per_file[2]
                a = np.percentile(score, min(a_,99))
                arr_max = np.where(score > a)[0]

                #
                per_file_ = [per_file[0][i] for i in arr_max]
                num_data = len(per_file_)
                per_file = [per_file_, np.zeros(num_data), np.full(num_data, 0.01)]
                list_player = [scoring] * (getAgentSize()-1) + [Ann_rdb]
                kq, per_file = normal_main(list_player, 5000, per_file)
                score1 = per_file[1]/per_file[2]
                list_player = [test_data] + [Ann_rdb] * (getAgentSize()-1)
                score2 = np.zeros(score1.shape)
                for i in range(len(per_file[0])):
                    data = per_file[0][i]
                    kq, temp = normal_main(list_player, 1000, data)
                    score2[i] = kq[0]/1000
                
                total_score = score1 + score2
                best_idx = total_score.argmax()
                cur_best_fnn_data = per_file[0][best_idx][0]
                np.save(f'{path_save_player}Ahih1st_FNN.npy', cur_best_fnn_data)