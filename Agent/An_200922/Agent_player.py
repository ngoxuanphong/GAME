import numpy as np
import os
import sys
from setup import game_name,time_run_game
sys.path.append(os.path.abspath(f"base/{game_name}"))
from env import *

player = 'An_200922'
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

def SoftPlus(x:np.ndarray):
    x_ = np.where(np.abs(x)>LOG_INF-1, x, np.log(1+np.e**(x)))
    return x_/np.max(x_)

def Gaussian(x:np.ndarray):
    return np.e**(-np.where(np.abs(x)>SQRT_LOG_INF, np.sign(x)*SQRT_LOG_INF, x)**2)

activation_function = [Identity, BinaryStep, Sigmoid, SignStep, Tanh, ReLU, SoftPlus, Gaussian]

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

    action_max_idx = np.random.choice(arr_max)
    return list_action[action_max_idx]

def generate_fnn(p_state, temp_file, per_file):
    if len(temp_file) < 2:
        temp_file = []
        len_state = len(p_state)
        total_action = getActionSize()
        number_layer = 3
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

def generate_g1(p_state, temp_file, per_file):
    list_action = getValidActions(p_state)
    list_action = np.where(list_action == 1)[0]
    action = np.random.choice(list_action)

    check_win = getReward(p_state)
    if check_win != -1:
        if check_win == 1:
            for i in temp_file[1:]:
                per_file[i] = 0.142625 + 0.857375*per_file[i]
        elif temp_file[0] == 0:
            temp_file[0] = 1
            for i in temp_file[1:]:
                per_file[i] *= 0.95
    
    temp_file.append(action)
    return action, temp_file, per_file

def generate_bnn(p_state, temp_file, per_file):
    if type(per_file) == list and len(per_file) == 0:
        per_file = np.full((len(p_state), getActionSize()), 0.5)
    
    list_action = getValidActions(p_state)
    list_action = np.where(list_action == 1)[0]
    action = np.random.choice(list_action)

    check_win = getReward(p_state)
    if check_win != -1:
        if check_win == 1:
            for pair in temp_file[1:]:
                res_arr = np.nan_to_num(pair[0] @ per_file)
                max_x = np.abs(res_arr).max()+1e-9
                if max_x != 0:
                    res_arr = res_arr/max_x
                
                am_inc = (1-res_arr[pair[1]])*0.142625*max_x
                if am_inc != 0:
                    a = p_state*per_file[:, pair[1]]
                    sum_square = np.sum(a**2)+1e-9
                    b = np.where(p_state<1e-9, 1e-9, p_state)
                    per_file[:, pair[1]] += a**2/sum_square*am_inc/b
                
                if np.min(per_file) < 0:
                    temp_abc = np.abs(np.min(per_file))
                    per_file = np.nan_to_num(per_file+temp_abc)

                per_file = per_file/np.abs(per_file).max()
        
        elif temp_file[0] == 0:
            temp_file[0] == 1
            for pair in temp_file[1:]:
                res_arr = np.nan_to_num(pair[0] @ per_file)
                max_x = np.abs(res_arr).max()+1e-9
                if max_x != 0:
                    res_arr = res_arr/max_x
                
                am_dec = (res_arr[pair[1]])*0.05*max_x
                if am_dec != 0:
                    a = p_state*per_file[:, pair[1]]
                    sum_square = np.sum(a**2)+1e-9
                    b = np.where(p_state<1e-9, 1e-9, p_state)
                    per_file[:, pair[1]] -= a**2/sum_square*am_dec/b
                
                if np.min(per_file) < 0:
                    temp_abc = np.abs(np.min(per_file))
                    per_file = np.nan_to_num(per_file+temp_abc)

                per_file = per_file/np.abs(per_file).max()
    
    temp_file.append([p_state, action])
    return action, temp_file, per_file

def test(p_state, temp_file, per_file):
    if len(temp_file) < 2:
        temp_file = np.load(f'{path_save_player}Ahih1st.npy', allow_pickle=True)
    
    list_action = getValidActions(p_state)
    list_action = np.where(list_action == 1)[0]
    if temp_file[1] == 0: # fnn
        action = Ann_neural_network(p_state, temp_file[0], list_action)
        return action, temp_file, per_file
    if temp_file[1] == 1: # sg
        if len(temp_file) < 3:
            temp_file.append(temp_file[0][0]/temp_file[0][1])

        res_arr = temp_file[2][list_action]
        a = np.max(res_arr)
        if a >= 0:
            arr_max = np.where(res_arr >= 0.99*a)[0]
        else:
            arr_max = np.where(res_arr >= 1.01*a)[0]

        action_max_idx = np.random.choice(arr_max)
        return list_action[action_max_idx], temp_file, per_file
    if temp_file[1] == 2: # g1
        res_arr = temp_file[0][list_action]
        a = np.max(res_arr)
        if a >= 0:
            arr_max = np.where(res_arr >= 0.99*a)[0]
        else:
            arr_max = np.where(res_arr >= 1.01*a)[0]

        action_max_idx = np.random.choice(arr_max)
        return list_action[action_max_idx], temp_file, per_file
    if temp_file[1] == 3: # bnn
        res_ = p_state @ temp_file[0]
        res_arr = res_[list_action]
        a = np.max(res_arr)
        if a >= 0:
            arr_max = np.where(res_arr >= 0.99*a)[0]
        else:
            arr_max = np.where(res_arr >= 1.01*a)[0]

        action_max_idx = np.random.choice(arr_max)
        return list_action[action_max_idx], temp_file, per_file

def test_data(p_state, temp_file, per_file):
    list_action = getValidActions(p_state)
    list_action = np.where(list_action == 1)[0]
    if per_file[1] == 0: # fnn
        action = Ann_neural_network(p_state, per_file[0], list_action)
        return action, temp_file, per_file
    if per_file[1] == 1: # sg
        if len(per_file) < 3:
            per_file.append(per_file[0][0]/per_file[0][1])

        res_arr = per_file[2][list_action]
        a = np.max(res_arr)
        if a >= 0:
            arr_max = np.where(res_arr >= 0.99*a)[0]
        else:
            arr_max = np.where(res_arr >= 1.01*a)[0]

        action_max_idx = np.random.choice(arr_max)
        return list_action[action_max_idx], temp_file, per_file
    if per_file[1] == 2: # g1
        res_arr = per_file[0][list_action]
        a = np.max(res_arr)
        if a >= 0:
            arr_max = np.where(res_arr >= 0.99*a)[0]
        else:
            arr_max = np.where(res_arr >= 1.01*a)[0]

        action_max_idx = np.random.choice(arr_max)
        return list_action[action_max_idx], temp_file, per_file
    if per_file[1] == 3: # bnn
        res_ = p_state @ per_file[0]
        res_arr = res_[list_action]
        a = np.max(res_arr)
        if a >= 0:
            arr_max = np.where(res_arr >= 0.99*a)[0]
        else:
            arr_max = np.where(res_arr >= 1.01*a)[0]

        action_max_idx = np.random.choice(arr_max)
        return list_action[action_max_idx], temp_file, per_file

def test_sg(p_state, temp_file, per_file):
    if len(temp_file) < 2:
        temp_file = np.load(f'{path_save_player}Ahih1st_SG.npy', allow_pickle=True)
        temp_file = [temp_file[0]/temp_file[1], 0]
    
    list_action = getValidActions(p_state)
    list_action = np.where(list_action == 1)[0]
    res_arr = temp_file[0][list_action]
    action_idx = res_arr.argmax()
    return list_action[action_idx], temp_file, per_file

def test_g1(p_state, temp_file, per_file):
    if len(temp_file) < 2:
        temp_file = [np.load(f'{path_save_player}Ahih1st_G1.npy', allow_pickle=True), 0]
    
    list_action = getValidActions(p_state)
    list_action = np.where(list_action == 1)[0]
    res_arr = temp_file[0][list_action]
    action_idx = res_arr.argmax()
    return list_action[action_idx], temp_file, per_file

def test_bnn(p_state, temp_file, per_file):
    if len(temp_file) < 2:
        temp_file = [np.load(f'{path_save_player}Ahih1st_BNN.npy', allow_pickle=True), 0]
    
    list_action = getValidActions(p_state)
    list_action = np.where(list_action == 1)[0]
    res_arr = p_state @ temp_file[0]
    res = res_arr[list_action]
    action_idx = res.argmax()
    return list_action[action_idx], temp_file, per_file

def test_fnn(p_state, temp_file, per_file):
    if len(temp_file) < 2:
        temp_file = np.load(f'{path_save_player}Ahih1st_FNN.npy', allow_pickle=True)
    
    list_action = getValidActions(p_state)
    list_action = np.where(list_action == 1)[0]
    action = Ann_neural_network(p_state, temp_file, list_action)
    return action, temp_file, per_file

def scoring(p_state, temp_file, per_file):
    if len(temp_file) < 2:
        k = np.random.randint(0, len(per_file[0]))
        temp_file = per_file[0][k] + [k]
        per_file[2][k] += 1
    
    list_action = getValidActions(p_state)
    list_action = np.where(list_action == 1)[0]
    if getReward(p_state) == 1:
        per_file[1][temp_file[2]] += 1

    if temp_file[1] == 0: # fnn
        action = Ann_neural_network(p_state, temp_file[0], list_action)
        return action, temp_file, per_file
    if temp_file[1] == 1: # sg
        if len(temp_file) < 4:
            temp_file.append(temp_file[0][0]/temp_file[0][1])

        res_arr = temp_file[3][list_action]
        a = np.max(res_arr)
        if a >= 0:
            arr_max = np.where(res_arr >= 0.99*a)[0]
        else:
            arr_max = np.where(res_arr >= 1.01*a)[0]

        action_max_idx = np.random.choice(arr_max)
        return list_action[action_max_idx], temp_file, per_file
    if temp_file[1] == 2: # g1
        res_arr = temp_file[0][list_action]
        a = np.max(res_arr)
        if a >= 0:
            arr_max = np.where(res_arr >= 0.99*a)[0]
        else:
            arr_max = np.where(res_arr >= 1.01*a)[0]

        action_max_idx = np.random.choice(arr_max)
        return list_action[action_max_idx], temp_file, per_file
    if temp_file[1] == 3: # bnn
        res_ = p_state @ temp_file[0]
        res_arr = res_[list_action]
        a = np.max(res_arr)
        if a >= 0:
            arr_max = np.where(res_arr >= 0.99*a)[0]
        else:
            arr_max = np.where(res_arr >= 1.01*a)[0]

        action_max_idx = np.random.choice(arr_max)
        return list_action[action_max_idx], temp_file, per_file

def train(num_cycle=1):
    # sg
    try:
        per_file = np.load(f'{path_save_player}Ahih1st_SG.npy', allow_pickle=True)
    except:
        per_file = [np.zeros(getActionSize()), np.full(getActionSize(), 0.01)]
    
    list_player = [generate_sg] * getAgentSize()
    kq, per_file = normal_main(list_player, 1000, per_file)
    np.save(f'{path_save_player}Ahih1st_SG.npy', per_file)

    # g1
    try:
        per_file = np.load(f'{path_save_player}Ahih1st_G1.npy', allow_pickle=True)
    except:
        per_file = np.full(getActionSize(), 0.5)
    
    list_player = [generate_g1] * getAgentSize()
    kq, per_file = normal_main(list_player, 1000, per_file)
    np.save(f'{path_save_player}Ahih1st_G1.npy', per_file)

    # bnn
    try:
        per_file = np.load(f'{path_save_player}Ahih1st_BNN.npy', allow_pickle=True)
    except:
        per_file = []

    list_player = [generate_bnn] * getAgentSize()
    kq, per_file = normal_main(list_player, 1000, per_file)
    np.save(f'{path_save_player}Ahih1st_BNN.npy', per_file)

    for _ in range(num_cycle):
        # sg
        data_sg = np.load(f'{path_save_player}Ahih1st_SG.npy', allow_pickle=True)
        per_file = data_sg
        
        list_player = [generate_sg] * getAgentSize()
        kq, per_file = normal_main(list_player, 1000, per_file)
        np.save(f'{path_save_player}Ahih1st_SG.npy', per_file)
        data_sg = np.load(f'{path_save_player}Ahih1st_SG.npy', allow_pickle=True)

        # g1
        data_g1 = np.load(f'{path_save_player}Ahih1st_G1.npy', allow_pickle=True)
        per_file = data_g1
        
        list_player = [generate_g1] * getAgentSize()
        kq, per_file = normal_main(list_player, 1000, per_file)
        np.save(f'{path_save_player}Ahih1st_G1.npy', per_file)
        data_g1 = np.load(f'{path_save_player}Ahih1st_G1.npy', allow_pickle=True)

        # bnn
        data_bnn = np.load(f'{path_save_player}Ahih1st_BNN.npy', allow_pickle=True)
        per_file = data_bnn

        list_player = [generate_bnn] * getAgentSize()
        kq, per_file = normal_main(list_player, 1000, per_file)
        np.save(f'{path_save_player}Ahih1st_BNN.npy', per_file)
        data_bnn = np.load(f'{path_save_player}Ahih1st_BNN.npy', allow_pickle=True)

        # fnn
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

        list_player = [generate_fnn] + [test_g1] + [Ann_rdb] * (getAgentSize() - 2)
        kq, per_file = normal_main(list_player, 1000, per_file)

        list_player = [generate_fnn] + [test_bnn] + [Ann_rdb] * (getAgentSize() - 2)
        kq, per_file = normal_main(list_player, 1000, per_file)

        list_player = [generate_fnn] + [Ann_rdb] * (getAgentSize() - 1)
        kq, per_file = normal_main(list_player, 1000, per_file)

        temp = per_file.copy()
        per_file = [[data, 0] for data in temp]
        per_file.append([data_sg, 1])
        per_file.append([data_g1, 2])
        per_file.append([data_bnn, 3])

        #########################
        num_data = len(per_file)
        a_ = (1-12/num_data)*100
        per_file = [per_file, np.zeros(num_data), np.full(num_data, 0.01)]
        list_player = [scoring] * (getAgentSize()-1) + [Ann_rdb]
        kq, per_file = normal_main(list_player, 20000, per_file)
        score = per_file[1]/per_file[2]
        a = np.percentile(score, a_)
        arr_max = np.where(score > a)[0]

        #
        per_file_ = [per_file[0][i] for i in arr_max]
        num_data = len(per_file_)
        per_file = [per_file_, np.zeros(num_data), np.full(num_data, 0.01)]
        list_player = [scoring] * (getAgentSize()-1) + [Ann_rdb]
        kq, per_file = normal_main(list_player, 10000, per_file)
        score1 = per_file[1]/per_file[2]
        list_player = [test_data] + [Ann_rdb] * (getAgentSize()-1)
        score2 = np.zeros(score1.shape)
        for i in range(len(per_file[0])):
            data = per_file[0][i]
            kq, temp = normal_main(list_player, 1000, data)
            score2[i] = kq[0]/500
        
        total_score = score1 + score2
        best_idx = total_score.argmax()
        cur_best_data = per_file[0][best_idx]
        np.save(f'{path_save_player}Ahih1st.npy', cur_best_data)

        temp = [per_file[0][i] for i in range(len(per_file[0])) if per_file[0][i][1] == 0]
        temp1 = np.array([total_score[i] for i in range(len(per_file[0])) if per_file[0][i][1] == 0])
        best_idx = temp1.argmax()
        cur_best_fnn_data = temp[best_idx][0]
        np.save(f'{path_save_player}Ahih1st_FNN.npy', cur_best_fnn_data)