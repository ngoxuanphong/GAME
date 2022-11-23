from system.mainFunc import dict_game_for_player, load_data_per2    
game_name_ = 'Splendor'
import random
import numpy as np
from setup import game_name,time_run_game
from numba import jit, njit, prange
import warnings
from numba.typed import List
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning,NumbaExperimentalFeatureWarning, NumbaWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)
warnings.simplefilter('ignore', category=NumbaWarning)

##########################################################

normal_cards_infor = np.array([[0, 2, 2, 2, 0, 0, 0], [0, 2, 3, 0, 0, 0, 0], [0, 2, 1, 1, 0, 2, 1], [0, 2, 0, 1, 0, 0, 2], [0, 2, 0, 3, 1, 0, 1], [0, 2, 1, 1, 0, 1, 1], [1, 2, 0, 0, 0, 4, 0], [0, 2, 2, 1, 0, 2, 0], [0, 1, 2, 0, 2, 0, 1], [0, 1, 0, 0, 2, 2, 0], [0, 1, 1, 0, 1, 1, 1], [0, 1, 2, 0, 1, 1, 1], [0, 1, 1, 1, 3, 0, 0], [0, 1, 0, 0, 0, 2, 1], [0, 1, 0, 0, 0, 3, 0], [1, 1, 4, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 4], [0, 0, 0, 0, 0, 0, 3], [0, 0, 0, 1, 1, 1, 2], [0, 0, 0, 0, 1, 2, 2], [0, 0, 1, 0, 0, 3, 1], [0, 0, 2, 0, 0, 0, 2], [0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 2, 1, 0, 0], [0, 4, 0, 2, 2, 1, 0], [0, 4, 1, 1, 2, 1, 0], [0, 4, 0, 1, 0, 1, 3], [1, 4, 0, 0, 4, 0, 0], [0, 4, 0, 2, 0, 2, 0], [0, 4, 2, 0, 0, 1, 0], [0, 4, 1, 1, 1, 1, 0], [0, 4, 0, 3, 0, 0, 0], [0, 3, 1, 0, 2, 0, 0], [0, 3, 1, 1, 1, 0, 1], [1, 3, 0, 4, 0, 0, 0], [0, 3, 1, 2, 0, 0, 2], [0, 3, 0, 0, 3, 0, 0], [0, 3, 0, 0, 2, 0, 2], [0, 3, 3, 0, 1, 1, 0], [0, 3, 1, 2, 1, 0, 1], [1, 2, 0, 3, 0, 2, 2], [2, 2, 0, 2, 0, 1, 4], [1, 2, 3, 0, 2, 0, 3], [2, 2, 0, 5, 3, 0, 0], [2, 2, 0, 0, 5, 0, 0], [3, 2, 0, 0, 6, 0, 0], [3, 1, 0, 6, 0, 0, 0], [2, 1, 1, 0, 0, 4, 2], [2, 1, 0, 5, 0, 0, 0], [2, 1, 0, 3, 0, 0, 5], [1, 1, 0, 2, 3, 3, 0], [1, 1, 3, 2, 2, 0, 0], [3, 0, 6, 0, 0, 0, 0], [2, 0, 0, 0, 0, 5, 3], [2, 0, 0, 0, 0, 5, 0], [2, 0, 0, 4, 2, 0, 1], [1, 0, 2, 3, 0, 3, 0], [1, 0, 2, 0, 0, 3, 2], [3, 4, 0, 0, 0, 0, 6], [2, 4, 5, 0, 0, 3, 0], [2, 4, 5, 0, 0, 0, 0], [1, 4, 3, 3, 0, 0, 2], [1, 4, 2, 0, 3, 2, 0], [2, 4, 4, 0, 1, 2, 0], [1, 3, 0, 2, 2, 0, 3], [1, 3, 0, 0, 3, 2, 3], [2, 3, 2, 1, 4, 0, 0], [2, 3, 3, 0, 5, 0, 0], [2, 3, 0, 0, 0, 0, 5], [3, 3, 0, 0, 0, 6, 0], [4, 2, 0, 7, 0, 0, 0], [4, 2, 0, 6, 3, 0, 3], [5, 2, 0, 7, 3, 0, 0], [3, 2, 3, 3, 0, 3, 5], [3, 1, 3, 0, 3, 5, 3], [4, 1, 0, 0, 0, 0, 7], [5, 1, 0, 3, 0, 0, 7], [4, 1, 0, 3, 0, 3, 6], [3, 0, 0, 5, 3, 3, 3], [4, 0, 0, 0, 7, 0, 0], [5, 0, 3, 0, 7, 0, 0], [4, 0, 3, 3, 6, 0, 0], [5, 4, 0, 0, 0, 7, 3], [3, 4, 5, 3, 3, 3, 0], [4, 4, 0, 0, 0, 7, 0], [4, 4, 3, 0, 0, 6, 3], [3, 3, 3, 3, 5, 0, 3], [5, 3, 7, 0, 0, 3, 0], [4, 3, 6, 0, 3, 3, 0], [4, 3, 7, 0, 0, 0, 0]])
noble_cards_infor = np.array([[0, 4, 4, 0, 0], [3, 0, 3, 3, 0], [3, 3, 3, 0, 0], [3, 0, 0, 3, 3], [0, 3, 0, 3, 3], [4, 0, 4, 0, 0], [4, 0, 0, 4, 0], [0, 3, 3, 0, 3], [0, 4, 0, 0, 4], [0, 0, 0, 4, 4]])


@njit
def getActionSize():
    return 198

@njit
def getAgentSize():
    return 4

@njit()
def getStateSize():
    return 165

@njit
def check_buy_card(p_state, card_id):
    self_st = p_state[106:112]
    self_st_const = p_state[112:117]
    card_price = normal_cards_infor[card_id][-5:]
    if np.sum((card_price>(self_st[:5]+self_st_const))*(card_price-self_st[:5]-self_st_const)) <= self_st[5]:
        return True

    return False


@njit
def getValidActions(player_state_origin:np.int64):
    list_action_return = np.zeros(198)
    p_state = player_state_origin.copy()
    p_state = p_state.astype(np.int64)
    phase = p_state[160] # Pha
    normal_cards = p_state[:90] # Trạng thái các thẻ thường
    b_stocks = p_state[100:106] # Nguyên liệu trên bàn chơi
    self_st = p_state[106:112] # Nguyên liệu của người chơi
    cards_check_buy = [i_ for i_ in range(90) if normal_cards[i_]==-1 or normal_cards[i_] == 5]

    if phase == 0: # Lựa chọn kiểu hành động
        if np.sum(b_stocks[:5]) != 0:
            # list_action.append(1) # Lấy nguyên liệu
            list_action_return[1] = 1
        else: 
            list_action_return[0] = 1
            # list_action.append(0) # Bỏ lượt

        if np.count_nonzero(normal_cards==-1) < 3:
            # list_action.append(2) # Úp thẻ
            list_action_return[2] = 1

        for card_id in cards_check_buy:
            if check_buy_card(p_state, card_id):
                # list_action.append(3) # Mua thẻ
                list_action_return[3] = 1
                break
    
    elif phase == 1: # Lấy nguyên liệu
        taken = p_state[155:160]
        s_taken = np.sum(taken)
        temp_ = [i_+4 for i_ in range(5) if b_stocks[i_] != 0]
        if s_taken == 0:
            # list_action += temp_
            list_action_return[np.array(temp_)] = 1
        elif s_taken == 1:
            s_ = np.where(taken==1)[0][0]
            if b_stocks[s_] >= 3: # Có thể lấy double
                # list_action += temp_
                list_action_return[np.array(temp_)] = 1
            else:
                if (s_+4) in temp_:
                    temp_.remove(s_+4)

                # list_action += temp_
                list_action_return[np.array(temp_)] = 1
        else:
            lst_s_ = np.where(taken==1)[0]
            for s_ in lst_s_:
                if (s_+4) in temp_:
                    temp_.remove(s_+4)
            
            # list_action += temp_
            list_action_return[np.array(temp_)] = 1

        if np.sum(self_st) >= 10:
            # list_action.append(0)
            list_action_return[0] = 1
    
    elif phase == 2: # Úp thẻ
        temp_ = [i_+9 for i_ in range(90) if normal_cards[i_]==5]
        # list_action += temp_
        list_action_return[np.array(temp_)] = 1
        if p_state[161] == 1:
            # list_action.append(99)
            list_action_return[99] = 1
        if p_state[162] == 1:
            # list_action.append(100)
            list_action_return[100] = 1
        if p_state[163] == 1:
            # list_action.append(101)
            list_action_return[101] = 1

    elif phase == 3: # Mua thẻ
        for card_id in cards_check_buy:
            if check_buy_card(p_state, card_id):
                # list_action.append(card_id+102)
                list_action_return[card_id+102] = 1
    
    else: # Pha trả nguyên liệu
        # list_action += [i_+192 for i_ in range(6) if self_st[i_] != 0]
        list_action_return[np.array([i_+192 for i_ in range(6) if self_st[i_] != 0])] = 1

    return list_action_return

@njit()
def data_to_layer_NhatAnh_130922(state,data0, data1):
    state = np.dot(state,data0)
    state *= state > 0
    active = state > 0
    layer1 = data1.flatten() * active
    return layer1

@njit()
def test2_NhatAnh_130922(state,file_per_2):
    state = state[:-1]
    layer = np.zeros(getActionSize())
    for id in range(len(file_per_2[0])):
        layer += data_to_layer_NhatAnh_130922(state,file_per_2[0][id], file_per_2[1][id])
    base = np.zeros(getActionSize())
    actions = getValidActions(state)
    actions = np.where(actions == 1)[0]
    for act in actions:
        base[act] = 1
    layer *= base
    base += layer
    action = np.random.choice(np.where(base == np.max(base))[0])
    return action

###########################################################

@njit()
def basic_act_NhatAnh_200922(state,base):
    actions = getValidActions(state)
    actions = np.where(actions == 1)[0]
    for act in base:
        if act in actions:
            return act
    ind = np.random.randint(len(actions))
    action = actions[ind]
    return action

@njit()
def test2_NhatAnh_200922(state,file_per_2):
    state = state[:-1]
    action = basic_act_NhatAnh_200922(state,file_per_2)
    return action

###########################################################

@njit()
def advance_act_NhatAnh_270922(state,data):
    for id in range(len(data[1])):
        x = data[1][id].reshape(len(data[1][id]), 1)
        mt = np.dot(state,x)
        if mt[0] <= 0:
            action = basic_act_NhatAnh_200922(state,data[0][id-1])
            return int(action)
        else:
            action = basic_act_NhatAnh_200922(state,data[0][id])
            return int(action)
    return np.random.choice(np.where(getValidActions(state) == 1)[0])

@njit()
def test2_NhatAnh_270922(state, file_per_2):
    state = state[:-1]
    action = advance_act_NhatAnh_270922(state,file_per_2)
    return action

#################################################################
#################################################################
#################################################################
@njit()
def _sigmoid_khanh_130922_(x):
    sig = 1 / (1 + np.exp(-x))
    return sig

@njit()
def _silu_khanh_130922_(x, theda = 1.0):
    return x * _sigmoid_khanh_130922_(theda *x)

@njit()
def neural_network_khanh_130922(play_state, file_temp0, file_temp1, file_temp2):
    if 55 < len(play_state) < 70 or len(play_state) > 250 : # TLMN , TLMN_v2 , CENTURY
        matran1 = np.dot(play_state,file_temp0)
        matran1 = 1 / (1 + np.exp(-matran1))
        matran21 = np.dot(matran1,file_temp1)
        matran21 *= (matran21 > 0)
        matran2 = np.dot(matran21, file_temp2)
        return matran2    
    elif 120 <len(play_state)  < 170:# SPLENDOr SPlendor_view_only
        matrix1 = np.dot(play_state, file_temp0)
        matrixRL1 = 1 / (1 + np.exp(-matrix1))
        matrix2 = np.dot(matrixRL1, file_temp1)
        matrixRL2 = 1 / (1 + np.exp(-matrix2))
        all_action_val = np.dot(matrixRL2, file_temp2 )
        return all_action_val  
    elif 170 < len(play_state) < 250  : #SHERIFF 
        matran1 = np.dot(play_state, file_temp0)
        matran1 = _silu_khanh_130922_(matran1, theda = 1.0)
        matran2 = np.dot(matran1, file_temp1)
        return matran2
    else :#SUSHIGO-main, MACHIKOR0
        matran1 = np.dot(play_state, file_temp0)
        matran1 *= (matran1 > 0)
        matran2 = np.dot(matran1, file_temp1)
        return matran2

@njit()
def neural_network_khanh_130922_2(play_state, file_temp0, file_temp1):
    if 170 < len(play_state) < 250  : #SHERIFF 
        matran1 = np.dot(play_state, file_temp0)
        matran1 = _silu_khanh_130922_(matran1, theda = 1.0)
        matran2 = np.dot(matran1, file_temp1)
        return matran2
    else :#SUSHIGO-main, MACHIKOR0
        matran1 = np.dot(play_state, file_temp0)
        matran1 *= (matran1 > 0)
        matran2 = np.dot(matran1, file_temp1)
        return matran2

@njit()
def test2_Khanh_130922(play_state, file_per_2):
    play_state = play_state[:-1]
    a = getValidActions(play_state)
    a = np.where(a == 1)[0]
    if len(file_per_2) == 3:
        matran2 = neural_network_khanh_130922(play_state, file_per_2[0], file_per_2[1], file_per_2[2])
    else:
        matran2 = neural_network_khanh_130922_2(play_state, file_per_2[0], file_per_2[1])
    max_ = 0
    action_max = a[random.randrange(len(a))]
    
    for act in a:
        if matran2[act] > max_:
            max_ = matran2[act]
            action_max = act
    return action_max

################################################################

@njit()
def relu6_khanh_200922(x):
    return np.minimum(np.maximum(0, x),6)

@njit()
def neural_network_khanh_200922(play_state, file_temp0, file_temp1, file_temp2):
    if 55 < len(play_state) < 70 or len(play_state) > 250 : # TLMN , TLMN_v2 , CENTURY
        matran1 = np.dot(play_state,file_temp0)
        matran1 = 1 / (1 + np.exp(-matran1))
        matran21 = np.dot(matran1,file_temp1)
        matran21 *= (matran21 > 0)
        matran2 = np.dot(matran21, file_temp2)
        return matran2    
    elif 120 <len(play_state)  < 170:# SPLENDOr SPlendor_view_only
        matrix1 = np.dot(play_state,file_temp0)
        matrixRL1 = 1 / (1 + np.exp(-matrix1))
        matrix2 = np.dot(matrixRL1,file_temp1)
        matrixRL2 = 1 / (1 + np.exp(-matrix2))
        all_action_val = np.dot(matrixRL2,file_temp2)
        return all_action_val       
    elif 170 < len(play_state) < 250  : #SHERIFF 
        matran1 = np.dot(play_state, file_temp0)
        matran1 = relu6_khanh_200922(matran1)
        matran2 = np.dot(matran1, file_temp1)
        return matran2
    else :#SUSHIGO-main, MACHIKOR0
        matran1 = np.dot(play_state, file_temp0)
        matran1 = np.tanh(matran1)
        matran2 = np.dot(matran1, file_temp1)
        return matran2
    
@njit()
def neural_network_khanh_200922_2(play_state, file_temp0, file_temp1):    
    if 170 < len(play_state) < 250  : #SHERIFF 
        matran1 = np.dot(play_state, file_temp0)
        matran1 = relu6_khanh_200922(matran1)
        matran2 = np.dot(matran1, file_temp1)
        return matran2
    else :#SUSHIGO-main, MACHIKOR0
        matran1 = np.dot(play_state, file_temp0)
        matran1 = np.tanh(matran1)
        matran2 = np.dot(matran1, file_temp1)
        return matran2

@njit()
def test2_Khanh_200922(play_state,file_per_2):
    play_state = play_state[:-1]
    a = getValidActions(play_state)
    a = np.where(a == 1)[0]
    if len(file_per_2) == 3:
        matran2 = neural_network_khanh_200922(play_state, file_per_2[0], file_per_2[1], file_per_2[2])
    else:
        matran2 = neural_network_khanh_200922_2(play_state, file_per_2[0], file_per_2[1])
    max_ = 0
    action_max = a[random.randrange(len(a))]
    for act in a:
        if matran2[act] > max_:
            max_ = matran2[act]
            action_max = act
    return action_max

#############################################################

@njit()
def neural_network_khanh_270922(play_state, file_temp0, file_temp1, file_temp2):
    if 55 < len(play_state) < 70 or len(play_state) > 250 : # TLMN , TLMN_v2 , CENTURY
        matran1 = np.dot(play_state,file_temp0)
        matran1 = 1 / (1 + np.exp(-matran1))
        matran21 = np.dot(matran1,file_temp1)
        matran21 *= (matran21 > 0)
        matran2 = np.dot(matran21, file_temp2)
        return matran2    
    elif 120 <len(play_state)  < 170:# SPLENDOr SPlendor_view_only
        matrix1 = np.dot(play_state,file_temp0)
        matrixRL1 = np.tanh(matrix1)
        matrix2 = np.dot(matrixRL1,file_temp1)
        matrixRL2 = relu6_khanh_200922(matrix2)
        all_action_val = np.dot(matrixRL2,file_temp2)
        return all_action_val       
    elif 170 < len(play_state) < 250  : #SHERIFF 
        matran1 = np.dot(play_state, file_temp0)
        matran1 = relu6_khanh_200922(matran1)
        matran2 = np.dot(matran1, file_temp1)
        return matran2
    else :#SUSHIGO-main, MACHIKOR0
        matran1 = np.dot(play_state, file_temp0)
        matran1 = np.tanh(matran1)
        matran2 = np.dot(matran1, file_temp1)
        return matran2

@njit()
def neural_network_khanh_270922_2(play_state, file_temp0, file_temp1):
    if 170 < len(play_state) < 250  : #SHERIFF 
        matran1 = np.dot(play_state, file_temp0)
        matran1 = relu6_khanh_200922(matran1)
        matran2 = np.dot(matran1, file_temp1)
        return matran2
    else :#SUSHIGO-main, MACHIKOR0
        matran1 = np.dot(play_state, file_temp0)
        matran1 = np.tanh(matran1)
        matran2 = np.dot(matran1, file_temp1)
        return matran2

@njit()
def test2_Khanh_270922(play_state,file_per_2):
    play_state = play_state[:-1]
    a = getValidActions(play_state)
    a = np.where(a == 1)[0]
    if len(file_per_2) == 3:
        matran2 = neural_network_khanh_270922(play_state, file_per_2[0], file_per_2[1], file_per_2[2])
    else:
        matran2 = neural_network_khanh_270922_2(play_state, file_per_2[0], file_per_2[1])
    max_ = 0
    action_max = a[random.randrange(len(a))]
    
    for act in a:
        if matran2[act] > max_:
            max_ = matran2[act]
            action_max = act
    return action_max

#################################################################
#################################################################
#################################################################
@njit()
def Identity_an_130922(x):
    return x

@njit()
def BinaryStep_an_130922(x):
    x[x>=0] = 1.0
    x[x<0] = 0.0
    return x

@njit()
def Sigmoid_an_130922(x):
    return 1.0 / (1.0 + np.e**(-x))

@njit()
def NegativePositiveStep_an_130922(x):
    x[x>=0] = 1.0
    x[x<0] = -1.0
    return x

@njit()
def Tanh_an_130922(x):
    return (np.e**(x) - np.e**(-x)) / (np.e**(x) + np.e**(-x))

@njit()
def ReLU_an_130922(x):
    return x * (x>0)

@njit()
def LeakyReLU_an_130922(x):
    x[x<0] *= 0.01
    return x

@njit()
def PReLU_an_130922(x, a=0.5):
    x[x<0] *= 0.5
    return x

@njit()
def Gaussian_an_130922(x):
    return np.e**(-x**2)

@njit()
def id_function_an_130922(id, res_mat, Identity_an_130922, BinaryStep_an_130922, Sigmoid_an_130922, NegativePositiveStep_an_130922, Tanh_an_130922, ReLU_an_130922, LeakyReLU_an_130922, PReLU_an_130922, Gaussian_an_130922):
    if id == 0: return Identity_an_130922(res_mat)
    elif id == 1: return BinaryStep_an_130922(res_mat)
    elif id == 2: return Sigmoid_an_130922(res_mat)
    elif id == 3: return NegativePositiveStep_an_130922(res_mat)
    elif id == 4: return Tanh_an_130922(res_mat)
    elif id == 5: return ReLU_an_130922(res_mat)
    elif id == 6: return LeakyReLU_an_130922(res_mat)
    elif id == 7: return PReLU_an_130922(res_mat)
    else: return Gaussian_an_130922(res_mat)

@njit()
def neural_network_an_130922(res_mat, data, list_action):
    for i in range(len(data)):
        if i % 2 == 0:
            res_mat = np.dot(res_mat, data[i])
            max_x = np.max(np.abs(res_mat))
            max_x_1 = max_x/25
            res_mat = res_mat / max_x_1
        else:
            id = int(data[i][0][0])
            res_mat = id_function_an_130922(id, res_mat, Identity_an_130922, BinaryStep_an_130922, Sigmoid_an_130922, NegativePositiveStep_an_130922, Tanh_an_130922, ReLU_an_130922, LeakyReLU_an_130922, PReLU_an_130922, Gaussian_an_130922)

    
    res_arr = res_mat[list_action]
    arr_max = np.where(res_arr == np.max(res_arr))[0]
    action_max_idx = np.random.choice(arr_max)
    return list_action[action_max_idx]

@njit()
def test2_An_130922(p_state, temp_file,  file_per_2):
    p_state = p_state[:-1]
    list_action = getValidActions(p_state)
    list_action = np.where(list_action == 1)[0]
    action = neural_network_an_130922(p_state, file_per_2, list_action)
    return action, temp_file,  file_per_2

############################################################
@njit()
def Identity_an_200922(x:np.ndarray):
    return x/np.abs(x).max()

@njit()
def BinaryStep_an_200922(x:np.ndarray):
    return np.where(x>=0, 1, 0).astype(np.float64)

@njit()
def Sigmoid_an_200922(x:np.ndarray):
    LOG_INF = 709.782712893384
    return 1/(1+np.e**(-np.where(np.abs(x)>LOG_INF, np.sign(x)*LOG_INF, x)))

@njit()
def SignStep_an_200922(x:np.ndarray):
    return np.sign(x)

@njit()
def Tanh_an_200922(x:np.ndarray):
    LOG_INF = 709.782712893384
    HALF_LOG_INF = 354.891356446692
    x_new = np.where(np.abs(x)>HALF_LOG_INF, np.sign(x)*HALF_LOG_INF, x)
    return (np.e**(2*x_new)-1)/(np.e**(2*x_new)+1)

@njit()
def ReLU_an_200922(x:np.ndarray):
    return np.where(x<0, 0, x)/np.max(x)

@njit()
def SoftPlus_an_200922(x:np.ndarray):
    LOG_INF = 709.782712893384
    x_ = np.where(np.abs(x)>LOG_INF-1, x, np.log(1+np.e**(x)))
    return x_/np.max(x_)

@njit()
def Gaussian_an_200922(x:np.ndarray):
    SQRT_LOG_INF = 18.838560360247595
    return np.e**(-np.where(np.abs(x)>SQRT_LOG_INF, np.sign(x)*SQRT_LOG_INF, x)**2)

@njit()
def id_function_an_200922(id, res_mat, Identity_an_200922, BinaryStep_an_200922, Sigmoid_an_200922, SignStep_an_200922, Tanh_an_200922, ReLU_an_200922, SoftPlus_an_200922, Gaussian_an_200922):
    if id == 0: return Identity_an_200922(res_mat)
    elif id == 1: return BinaryStep_an_200922(res_mat)
    elif id == 2: return Sigmoid_an_200922(res_mat)
    elif id == 3: return SignStep_an_200922(res_mat)
    elif id == 4: return Tanh_an_200922(res_mat)
    elif id == 5: return ReLU_an_200922(res_mat)
    elif id == 6: return SoftPlus_an_200922(res_mat)
    else: return Gaussian_an_200922(res_mat)

@njit()
def Ann_neural_network_an_200922(res_mat:np.ndarray, data, list_action):
    for i in range(len(data)//3):
        data3i = data[3*i]
        data3i1 = data[3*i+1].flatten()
        data3i2 = int(data[3*i+2][0][0])
        res_mat = np.dot(res_mat, data3i) + data3i1
        res_mat = id_function_an_200922(data3i2, res_mat, Identity_an_200922, BinaryStep_an_200922, Sigmoid_an_200922, SignStep_an_200922, Tanh_an_200922, ReLU_an_200922, SoftPlus_an_200922, Gaussian_an_200922)
    
    res_arr = res_mat[list_action]
    a = np.max(res_arr)
    if a >= 0:
        arr_max = np.where(res_arr >= 0.99*a)[0]
    else:
        arr_max = np.where(res_arr >= 1.01*a)[0]

    action_max_idx = np.random.choice(arr_max)
    return list_action[action_max_idx]

@njit()
def test2_An_200922(p_state, file_per_2):
    p_state = p_state[:-1]
    list_action = getValidActions(p_state)
    list_action = np.where(list_action == 1)[0]
    if len(file_per_2) == 2: 
        type_file_per_2 = int(file_per_2[1][0][0][0])
    else:
        type_file_per_2 = int(file_per_2[0][1][0][0])
    
    if type_file_per_2 == 0: # fnn
        action = Ann_neural_network_an_200922(p_state, file_per_2[0], list_action)
    else: # sg
            res_arr = file_per_2[0][2][0][list_action]
            a = np.max(res_arr)
            if a >= 0:
                arr_max = np.where(res_arr >= 0.99*a)[0]
            else:
                arr_max = np.where(res_arr >= 1.01*a)[0]
            action_max_idx = np.random.choice(arr_max)
            action = list_action[action_max_idx]

    return action


####################################################

@njit()
def Identity_an_270922(x:np.ndarray):
    return x/np.abs(x).max()

@njit()
def BinaryStep_an_270922(x:np.ndarray):
    return np.where(x>=0, 1, 0).astype(np.float64)

@njit()
def Sigmoid_an_270922(x:np.ndarray):
    LOG_INF = 709.782712893384
    return 1/(1+np.e**(-np.where(np.abs(x)>LOG_INF, np.sign(x)*LOG_INF, x)))

@njit()
def SignStep_an_270922(x:np.ndarray):
    return np.sign(x)

@njit()
def Tanh_an_270922(x:np.ndarray):
    HALF_LOG_INF = 354.891356446692
    x_new = np.where(np.abs(x)>HALF_LOG_INF, np.sign(x)*HALF_LOG_INF, x)
    return (np.e**(2*x_new)-1)/(np.e**(2*x_new)+1)

@njit()
def ReLU_an_270922(x:np.ndarray):
    return np.where(x<0, 0, x)/np.max(x)

@njit()
def LeakyReLU_an_270922(x:np.ndarray):
    x_new = np.where(x<0, 0.01*x, x)
    return x_new/np.abs(x_new).max()

@njit()
def PReLU_an_270922(x:np.ndarray):
    x_new = np.where(x<0, 0.5*x, x)
    return x_new/np.abs(x_new).max()

@njit()
def SoftPlus_an_270922(x:np.ndarray):
    LOG_INF = 709.782712893384
    x_new = np.where(np.abs(x)>LOG_INF-1e-9, x, np.log(1+np.e**(x)))
    return x_new/np.max(x_new)

@njit()
def Gaussian_an_270922(x:np.ndarray):
    SQRT_LOG_INF = 18.838560360247595
    return np.e**(-np.where(np.abs(x)>SQRT_LOG_INF, np.sign(x)*SQRT_LOG_INF, x)**2)

activation_function = [Identity_an_270922, BinaryStep_an_270922, Sigmoid_an_270922, SignStep_an_270922, Tanh_an_270922, ReLU_an_270922, LeakyReLU_an_270922, PReLU_an_270922, SoftPlus_an_270922, Gaussian_an_270922]

@njit()
def id_function_an_270922(id, res_mat, Identity_an_270922, BinaryStep_an_270922, Sigmoid_an_270922, SignStep_an_270922, Tanh_an_270922, ReLU_an_270922, LeakyReLU_an_270922, PReLU_an_270922, SoftPlus_an_270922, Gaussian_an_270922):
    if id == 0: return Identity_an_270922(res_mat)
    if id == 1: return BinaryStep_an_270922(res_mat)
    if id == 2: return Sigmoid_an_270922(res_mat)
    if id == 3: return SignStep_an_270922(res_mat)
    if id == 4: return Tanh_an_270922(res_mat)
    if id == 5: return ReLU_an_270922(res_mat)
    if id == 6: return LeakyReLU_an_270922(res_mat)
    if id == 7: return PReLU_an_270922(res_mat)
    if id == 8: return SoftPlus_an_270922(res_mat)
    else: return Gaussian_an_270922(res_mat)

@njit()
def Ann_neural_network_an_270922(res_mat:np.ndarray, data, list_action):
    for i in range(len(data)//3):
        data3i = data[3*i]
        data3i1 = data[3*i+1].flatten()
        data3i2 = int(data[3*i+2][0][0])
        res_mat = np.dot(res_mat, data3i) + data3i1
        res_mat = id_function_an_270922(data3i2, res_mat, Identity_an_270922, BinaryStep_an_270922, Sigmoid_an_270922, SignStep_an_270922, Tanh_an_270922, ReLU_an_270922, LeakyReLU_an_270922, PReLU_an_270922, SoftPlus_an_270922, Gaussian_an_270922)
    
    res_arr = res_mat[list_action]
    a = np.max(res_arr)
    if a >= 0:
        arr_max = np.where(res_arr >= 0.99*a)[0]
    else:
        arr_max = np.where(res_arr >= 1.01*a)[0]
    
    return list_action[np.random.choice(arr_max)]

@njit()
def test2_An_270922(p_state, file_per_2):
    p_state = p_state[:-1]
    list_action = getValidActions(p_state)
    list_action = np.where(list_action == 1)[0]
    if len(file_per_2) == 2: 
        type_file_per_2 = int(file_per_2[1][0][0][0])
    else:
        type_file_per_2 = int(file_per_2[0][1][0][0])
    if type_file_per_2 == 0:
        action = Ann_neural_network_an_270922(p_state, file_per_2[0], list_action)
        # return action, temp_file,  file_per_2
    else:

        res_arr = file_per_2[0][2][0][list_action]
        a = np.max(res_arr)
        arr_max = np.where(res_arr >= 0.99*a)[0]
        action = list_action[np.random.choice(arr_max)]
    return action


#################################################################
#################################################################
#################################################################

@njit()
def test2_Dat_130922(state,file_per_2):
    state = state[:-1]
    list_action = getValidActions(state)
    list_action = np.where(list_action == 1)[0]
    hidden1 = np.dot(state, file_per_2[0])
    hidden2 = hidden1 * (hidden1>0)
    values =  np.dot(hidden2, file_per_2[1])
    action = list_action[np.argmax(values[list_action])]
    return action


###############################################################
###############################################################
###############################################################
@njit()
def neural_network_hieu_130922(state, file_temp0, file_temp1, file_temp2, list_action):
    norm_state = state/np.linalg.norm(state, 1)
    norm_state = np.tanh(norm_state)                    #dạng tanh
    norm_action = np.zeros(getActionSize())
    norm_action[list_action] = 1
    norm_action = norm_action.reshape(1, getActionSize())
    matrixRL1 = np.dot(norm_state, file_temp0)
    matrixRL1 = matrixRL1*(matrixRL1 > 0)           #activation = relu
    matrixRL2 = np.dot(matrixRL1, file_temp1)
    matrixRL2 = 1 / (1 + np.exp(-matrixRL2))            #activation = sigmoid
    matrixRL3 = np.dot(matrixRL2, file_temp2)
    matrixRL3 = np.tanh(matrixRL3)              #activation = tanh
    result_val_action = matrixRL3*norm_action
    action_max = np.argmax(result_val_action)
    return action_max

@njit()
def test2_Hieu_130922(state, file_per_2):
    state = state[:-1]
    list_action = getValidActions(state)
    list_action = np.where(list_action == 1)[0]
    action = neural_network_hieu_130922(state, file_per_2[0], file_per_2[1], file_per_2[2], list_action)
    return action
#################################################################
@njit()
def agent_hieu_270922(state,file_temp,file_per):
    actions = getValidActions(state)
    actions = np.where(actions == 1)[0]
    action = np.random.choice(actions)
    file_per = (len(state),getActionSize())
    return action,file_temp,file_per

@njit()
def softmax_hieu_270922(X):
    expo = np.exp(X)
    return expo/np.sum(expo)

@njit()
def sigmoid_hieu_270922(X):
    return 1/(1+np.exp(-X))

@njit()
def tanh_hieu_270922(X):
    return np.tanh(X)

@njit()
def neural_network_hieu_270922(norm_state, file_temp0, file_temp1, file_temp2, list_action):
    norm_state = norm_state/np.linalg.norm(norm_state, 1)
    norm_state = softmax_hieu_270922(norm_state)
    norm_action = np.zeros(getActionSize())
    norm_action[list_action] = 1
    norm_action = norm_action.reshape(1, getActionSize())

    matrixRL1 = np.dot(norm_state, file_temp0)
    matrixRL1 = sigmoid_hieu_270922(matrixRL1)          

    matrixRL2 = np.dot(matrixRL1, file_temp1)
    matrixRL2 = tanh_hieu_270922(matrixRL2)         

    matrixRL3 = np.dot(matrixRL2, file_temp2)
    matrixRL3 = softmax_hieu_270922(matrixRL3)   

    result_val_action = matrixRL3*norm_action
    action_max = np.argmax(result_val_action)
    return action_max

@njit()
def test2_Hieu_270922(state, file_per_2):
    state = state[:-1]
    list_action = getValidActions(state)
    list_action = np.where(list_action == 1)[0]
    action = neural_network_hieu_270922(state, file_per_2[0], file_per_2[1], file_per_2[2], list_action)
    return action

######################################################################
######################################################################
######################################################################

@njit()
def file_temp_to_action_Phong_130922(state, file_temp):
    a = getValidActions(state)
    a = np.where(a == 1)[0]
    RELU = np.ones(len(state))
    matrix_new = np.dot(RELU,file_temp)
    list_val_action = matrix_new[a]
    action = a[np.argmax(list_val_action)]
    return action

@njit() 
def test2_Phong_130922(state,file_per_2):
    state = state[:-1]
    action = file_temp_to_action_Phong_130922(state, file_per_2)
    return action



@njit()
def random_Env(p_state):
    p_state = p_state[:-1]
    arr_action = getValidActions(p_state)
    arr_action = np.where(arr_action == 1)[0]
    act_idx = np.random.randint(0, len(arr_action))
    return arr_action[act_idx]









