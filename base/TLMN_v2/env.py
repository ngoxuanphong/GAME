from numba import njit
from numba.typed import List
import numpy as np
import random as rd
import random

from system.mainFunc import dict_game_for_player
from main import load_data_per2
game_name_ = 'TLMN_v2'
# from colorama import Fore, Style

# -------------------- NOPYTHON FUNCTIONS --------------------
@njit
def reset(e_state):
    temp = np.arange(52)
    np.random.shuffle(temp)
    # 0 -> 51: State các lá bài
    for i in range(4):
        e_state[temp[13*i:13*(i+1)]] = i
    
    # 52: Player_id
    # 57: Chủ bộ bài được đánh trên bàn
    # 58, 59: Kiểu bộ bài, điểm bộ bài
    # 60: Phase mấy: 0 là phase chọn kiểu bộ bài, 1 là phase chọn lá cao nhất
    # 61: Kiểu bộ bài đã chọn ở phase 0, mặc định là 0
    e_state[52:62] = 0
    # 53 -> 56 Tình trạng bỏ vòng
    e_state[53:57] = 1

@njit
def get_player_state(e_state):
    p_state = np.full(62,0)
    # 0 -> 51: Index các lá bài. 0 Là trên tay, -1 là đã đánh, 1 là của người chơi khác
    p_state[0:52] = e_state[0:52]
    p_state[np.where(e_state[0:52] == e_state[52])[0]] = 0
    p_state[np.where((e_state[0:52] != e_state[52]) & (e_state[0:52] != -1))[0]] = 1

    # 52, 53, 54: Tình trạng bỏ vòng
    temp = np.arange(4)
    p_state[52:55] = e_state[53+np.concatenate((temp[e_state[52]+1:4], temp[0:e_state[52]]), axis=0)]

    # 55, 56, 57: Số lá bài còn lại
    for i in range(1,4):
        p_state[54+i] = np.count_nonzero(e_state[0:52] == (e_state[52]+i)%4)
    
    # 58, 59: Kiểu bộ bài, Điểm bộ bài
    if e_state[52] == e_state[57]:
        p_state[58:60] = 0
    else:
        p_state[58:60] = e_state[58:60]
    
    # 60: Phase mấy: 0 là phase chọn kiểu bộ bài, 1 là phase chọn lá bài cao nhất
    # 61: Kiểu bộ bài đã chọn ở phase 0
    p_state[60:62] = e_state[60:62]

    return p_state.astype(np.float64)

@njit
def straight_subsequences(arr):
    arr_return = []
    n = len(arr)
    for k in range(3,12):
        if n < k:
            break
        
        for i in range(0, n-k+1):
            sub_arr = arr[i:i+k]
            if np.max(sub_arr) - np.min(sub_arr) == k-1:
                arr_return.append([k, sub_arr[k-1]])
        
    if len(arr_return) == 0:
        return np.full((0,2),0)
    else:
        return np.array(arr_return)

@njit
def hand_of_cards(arr_card):
    arr_return = []

    arr_return.append([0,0])

    for j in arr_card:
        arr_return.append([1,j])

    for i in range(13):
        temp = arr_card[arr_card//4 == i]
        for n in range(2,5):
            if len(temp) >= n and (i != 12 or n != 4):
                for j in temp[n-1:]:
                    arr_return.append([n,j])
    
    temp = np.unique(arr_card//4)
    arr_score = temp[temp < 12]
    arr_straight_subsequence = straight_subsequences(arr_score)
    for straight in arr_straight_subsequence:
        temp = arr_card[arr_card//4 == straight[1]]
        for j in temp:
            arr_return.append([straight[0]+2,j])
    
    arr_score = []
    for i in range(12):
        temp = arr_card[arr_card//4 == i]
        if len(temp) >= 2:
            arr_score.append(i)
    
    arr_straight_subsequence = straight_subsequences(np.array(arr_score))
    for straight in arr_straight_subsequence:
        if straight[0] <= 4:
            temp = arr_card[arr_card//4 == straight[1]]
            for j in temp[1:]:
                arr_return.append([straight[0]+11,j])

    return np.array(arr_return)

@njit
def get_list_action_old(player_state_origin:np.int64):
    p_state = player_state_origin.copy()
    p_state = p_state.astype(np.int64)
    arr_card = np.where(p_state[0:52] == 0)[0]
    arr_hand = hand_of_cards(arr_card)

    if len(arr_card) == 0:
        mask = (arr_hand[:,0] == 0)
    else:
        if p_state[58] == 0:
            mask = (arr_hand[:,0] != 0)
        else:
            if (p_state[58] >= 1 and p_state[58] <= 3) or (p_state[58] >= 5 and p_state[58] <= 13):
                if p_state[59] <= 47:
                    mask = ((arr_hand[:,0] == p_state[58]) & (arr_hand[:,1] > p_state[59])) | \
                            (arr_hand[:,0] == 0)
                else:
                    if p_state[58] == 1:
                        mask = ((arr_hand[:,0] == 1) & (arr_hand[:,1] > p_state[59])) | \
                                (arr_hand[:,0] == 4) | (arr_hand[:,0] == 14) | (arr_hand[:,0] == 15) | \
                                (arr_hand[:,0] == 0)
                    elif p_state[58] == 2:
                        mask = ((arr_hand[:,0] == 2) & (arr_hand[:,1] > p_state[59])) | \
                                (arr_hand[:,0] == 4) | (arr_hand[:,0] == 15) | \
                                (arr_hand[:,0] == 0)
                    else:
                        mask = (arr_hand[:,0] == 0)
            elif p_state[58] == 14:
                mask = ((arr_hand[:,0] == 14) & (arr_hand[:,1] > p_state[59])) | \
                        (arr_hand[:,0] == 4) | (arr_hand[:,0] == 15) | \
                        (arr_hand[:,0] == 0)
            elif p_state[58] == 4:
                mask = ((arr_hand[:,0] == 4) & (arr_hand[:,1] > p_state[59])) | \
                        (arr_hand[:,0] == 15) | \
                        (arr_hand[:,0] == 0)
            else:
                mask = ((arr_hand[:,0] == 15) & (arr_hand[:,1] > p_state[59])) | \
                        (arr_hand[:,0] == 0)
    
    possible_hands = arr_hand[mask,:]

    if p_state[60] == 0: # Phase chọn kiểu bộ bài
        return np.unique(possible_hands[:,0])
    else:
        mask_1 = possible_hands[:,0] == p_state[61]
        return possible_hands[mask_1,:][:,1] + 16

@njit
def get_list_action(player_state_origin:np.int64):
    list_action_return = np.zeros(68)
    p_state = player_state_origin.copy()
    p_state = p_state.astype(np.int64)
    arr_card = np.where(p_state[0:52] == 0)[0]
    arr_hand = hand_of_cards(arr_card)

    if len(arr_card) == 0:
        mask = (arr_hand[:,0] == 0)
    else:
        if p_state[58] == 0:
            mask = (arr_hand[:,0] != 0)
        else:
            if (p_state[58] >= 1 and p_state[58] <= 3) or (p_state[58] >= 5 and p_state[58] <= 13):
                if p_state[59] <= 47:
                    mask = ((arr_hand[:,0] == p_state[58]) & (arr_hand[:,1] > p_state[59])) | \
                            (arr_hand[:,0] == 0)
                else:
                    if p_state[58] == 1:
                        mask = ((arr_hand[:,0] == 1) & (arr_hand[:,1] > p_state[59])) | \
                                (arr_hand[:,0] == 4) | (arr_hand[:,0] == 14) | (arr_hand[:,0] == 15) | \
                                (arr_hand[:,0] == 0)
                    elif p_state[58] == 2:
                        mask = ((arr_hand[:,0] == 2) & (arr_hand[:,1] > p_state[59])) | \
                                (arr_hand[:,0] == 4) | (arr_hand[:,0] == 15) | \
                                (arr_hand[:,0] == 0)
                    else:
                        mask = (arr_hand[:,0] == 0)
            elif p_state[58] == 14:
                mask = ((arr_hand[:,0] == 14) & (arr_hand[:,1] > p_state[59])) | \
                        (arr_hand[:,0] == 4) | (arr_hand[:,0] == 15) | \
                        (arr_hand[:,0] == 0)
            elif p_state[58] == 4:
                mask = ((arr_hand[:,0] == 4) & (arr_hand[:,1] > p_state[59])) | \
                        (arr_hand[:,0] == 15) | \
                        (arr_hand[:,0] == 0)
            else:
                mask = ((arr_hand[:,0] == 15) & (arr_hand[:,1] > p_state[59])) | \
                        (arr_hand[:,0] == 0)
    
    possible_hands = arr_hand[mask,:]

    if p_state[60] == 0: # Phase chọn kiểu bộ bài
        # list_action = np.unique(possible_hands[:,0])
        list_action_return[np.unique(possible_hands[:,0])] = 1
        return list_action_return
    else:
        mask_1 = possible_hands[:,0] == p_state[61]
        # list_action = possible_hands[mask_1,:][:,1] + 16
        list_action_return[possible_hands[mask_1,:][:,1] + 16] = 1
        return list_action_return


@njit
def close_game(e_state):
    for i in range(4):
        if np.count_nonzero(e_state[0:52] == i) == 0:
            return i
    
    return -1

@njit
def amount_action():
    return 68

@njit
def amount_player():
    return 4

@njit
def check_victory(p_state):
    a = np.count_nonzero(p_state[0:52]==0)
    b = np.min(p_state[55:58])
    if a*b == 0:
        if a == 0:
            return 1
        else:
            return 0
    else:
        return -1

@njit
def step(action, e_state):
    p_state = get_player_state(e_state)
    list_action = get_list_action(p_state)
    if list_action[action] != 1:
        '''
        Action không hợp lệ
        '''
        print('Action không hợp lệ')
    
    else:
        if e_state[60] == 0: # Phase chọn kiểu bộ bài
            if action == 0: # Bỏ lượt, loại khỏi vòng, sang turn của người chơi tiếp theo
                e_state[53:57][e_state[52]] = 0 # Loại khỏi vòng
                for i in range(1,4): # Xác định người chơi tiếp theo
                    if e_state[53:57][(e_state[52]+i)%4] == 1:
                        e_state[52] = (e_state[52]+i)%4
                        break
            else:
                e_state[60] = 1 # Chuyển phase
                e_state[61] = action
        else:
            arr_card = np.where(p_state[0:52] == 0)[0]
            arr_card_in_hand = []
            hand = np.array([e_state[61], action-16])
            if hand[0] >= 1 and hand[0] <= 4:
                temp = arr_card[arr_card//4 == hand[1]//4]
                for j in temp[0:hand[0]-1]:
                    arr_card_in_hand.append(j)
                else:
                    arr_card_in_hand.append(hand[1])
            elif hand[0] >= 5 and hand[0] <= 13:
                last = hand[1]//4
                straight_len = hand[0] - 2
                for i in range(last-straight_len+1, last):
                    temp = arr_card[arr_card//4 == i]
                    arr_card_in_hand.append(temp[0])
                else:
                    arr_card_in_hand.append(hand[1])
            else:
                last = hand[1]//4
                straight_len = hand[0] - 11
                for i in range(last-straight_len+1, last):
                    temp = arr_card[arr_card//4 == i]
                    for j in temp[0:2]:
                        arr_card_in_hand.append(j)
                else:
                    temp = arr_card[arr_card//4 == last]
                    arr_card_in_hand.append(temp[0])
                    arr_card_in_hand.append(hand[1])
            
            e_state[np.array(arr_card_in_hand)] = -1

            e_state[57] = e_state[52]
            e_state[58:60] = hand
            e_state[60:62] = 0

            if hand[1] >= 48 or hand[0] == 4 or hand[0] == 14 or hand[0] == 15 or np.sum(e_state[53:57]) == 1:
                e_state[53:57] = 1
            
            for i in range(1,4):
                if e_state[53:57][(e_state[52]+i)%4] == 1:
                    e_state[52] = (e_state[52]+i)%4
                    break
            
            return np.array(arr_card_in_hand)

@njit
def check_env(e_state):
    for i in range(4):
        arr_card = np.where(e_state[0:52] == i)[0]

        temp = arr_card[arr_card//4 == 12]
        if len(temp) == 4:
            return False
        
        temp = np.unique(arr_card//4)
        arr_score = temp[temp < 12]
        if len(arr_score) == 12:
            return False
        
        arr_score = []
        for i in range(12):
            temp = arr_card[arr_card//4 == i]
            if len(temp) >= 2:
                arr_score.append(i)
        
        arr_straight_subsequence = straight_subsequences(np.array(arr_score))
        mask = (arr_straight_subsequence[:,0] >= 5)
        temp = arr_straight_subsequence[mask,:]
        if len(temp) > 0:
            return False
    
    return True

# --------------------  PYTHON FUNCTIONS  --------------------
def convert(arr_card):
    arr1 = arr_card // 4
    arr2 = arr_card % 4
    str1 = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2']
    str2 = ['Bích', 'Tép', 'Rô', 'Cơ']
    return [str1[arr1[i]] + ' ' + str2[arr2[i]] for i in range(len(arr_card))]

def convert_1(k):
    str1 = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2']
    str2 = ['Bích', 'Tép', 'Rô', 'Cơ']
    return str1[k//4] + ' ' + str2[k%4]

# def print_list_card(list_card):
#     for card in list_card:
#         temp = card.split(' ')
#         if temp[1] in ['Rô', 'Cơ']:
#             print(Fore.LIGHTWHITE_EX + temp[0] + Fore.LIGHTRED_EX, temp[1], end=' ')
#         else:
#             print(Fore.LIGHTWHITE_EX + temp[0] + Fore.LIGHTBLACK_EX, temp[1], end=' ')
    
#     print(Style.RESET_ALL)

# def print_player_cards(e_state):
#     for i in range(4):
#         temp = np.where(e_state[0:52] == i)[0]
#         temp = convert(temp)
#         print(Fore.LIGHTGREEN_EX + 'Player', i+1, ':', end=' ')
#         print_list_card(temp)

# def print_ingame(act, arr_card_in_hand, e_state):
#     print(Fore.LIGHTCYAN_EX + 'Chọn action' + Style.RESET_ALL, act, act-16, convert_1(act-16))
#     print(Fore.LIGHTCYAN_EX + 'Các lá bài trong action:' + Style.RESET_ALL, end=' ')
#     print_list_card(convert(arr_card_in_hand))
#     print('----------------------------------------------------------------------------------------------------')
#     print_player_cards(e_state)

def random_player(p_state, temp_file, per_file):
    arr_action = get_list_action(p_state)
    act_idx = np.random.randint(0, len(arr_action))
    return arr_action[act_idx], temp_file, per_file

# --------------------        MAIN        --------------------
def one_game(list_player, env, per_file):
    reset(env)
    while not check_env(env):
        reset(env)
    
    temp_file = [[0], [0], [0], [0]]
    while True:
        act, temp_file[env[52]], per_file = list_player[env[52]](get_player_state(env), temp_file[env[52]], per_file)
        arr_card_in_hand = step(act, env)
        if close_game(env) != -1:
            break
    
    winner = close_game(env)
    for i in range(4):
        env[52] = i
        act, temp_file[env[52]], per_file = list_player[env[52]](get_player_state(env), temp_file[env[52]], per_file)
    
    return winner, per_file

# def one_game_print(list_player, env, print_mode, per_file):
#     reset(env)
#     while not check_env(env):
#         reset(env)
    
#     if print_mode:
#         print_player_cards(env)
    
#     temp_file = [[0], [0], [0], [0]]
#     while True:
#         act, temp_file[env[52]], per_file = list_player[env[52]](get_player_state(env), temp_file[env[52]], per_file)
#         if print_mode and env[60] == 0:
#             print(Fore.LIGHTCYAN_EX + 'Lượt của player' + Style.RESET_ALL, env[52]+1, Fore.LIGHTCYAN_EX + '. Round state:' + Style.RESET_ALL, env[53:57])
#             print(get_list_action(get_player_state(env)), '. Hand type:', act)
#             if act == 0:
#                 flag = False
#             else:
#                 flag = True
        
#         arr_card_in_hand = step(act, env)
#         if print_mode and env[60] == 0 and flag:
#             print_ingame(act, arr_card_in_hand, env)
        
#         if close_game(env) != -1:
#             break
    
#     winner = close_game(env)
#     for i in range(4):
#         env[52] = i
#         act, temp_file[env[52]], per_file = list_player[env[52]](get_player_state(env), temp_file[env[52]], per_file)
    
#     return winner, per_file

def normal_main(list_player, num_game, per_file):
    if len(list_player) != 4:
        print('Game chỉ cho phép có đúng 4 người chơi')
        return [-1,-1,-1,-1], per_file
    
    env = np.full(62,0)
    count_win = [0,0,0,0]
    p_lst_idx = [0,1,2,3]
    for _n in range(num_game):
        rd.shuffle(p_lst_idx)
        winner, per_file = one_game(
            [list_player[p_lst_idx[0]], list_player[p_lst_idx[1]], list_player[p_lst_idx[2]], list_player[p_lst_idx[3]]], env, per_file
        )

        count_win[p_lst_idx[winner]] += 1
    
    return count_win, per_file

def n_games(list_player, num_game, print_mode):
    per_file = [0]
    if len(list_player) != 4:
        print('Game chỉ cho phép có đúng 4 người chơi')
        return [-1,-1,-1,-1], per_file

    env = np.full(62,0)
    count_win = [0,0,0,0]
    p_lst_idx = [0,1,2,3]
    for _n in range(num_game):
        # if _n % 100 == 0 and _n != 0:
        #     print(_n, count_win)
        
        rd.shuffle(p_lst_idx)
        winner, per_file = one_game(
            [list_player[p_lst_idx[0]], list_player[p_lst_idx[1]], list_player[p_lst_idx[2]], list_player[p_lst_idx[3]]], env, print_mode, per_file
        )

        count_win[p_lst_idx[winner]] += 1
    
    # print(num_game, count_win)
    
    return count_win, per_file


def one_game_2(list_player, env,  _file_per_2_):
    reset(env)
    while not check_env(env):
        reset(env)
    
    temp_file = [[0], [0], [0], [0]]
    while True:
        act, temp_file[env[52]],  _file_per_2_[env[52]] = list_player[env[52]](get_player_state(env), temp_file[env[52]],  _file_per_2_[env[52]])
        arr_card_in_hand = step(act, env)
        if close_game(env) != -1:
            break
    
    winner = close_game(env)
    for i in range(4):
        env[52] = i
        act, temp_file[env[52]],  _file_per_2_[env[52]] = list_player[env[52]](get_player_state(env), temp_file[env[52]],  _file_per_2_[env[52]])
    
    return winner,  _file_per_2_


def normal_main_2(list_player, num_game,  per_file_2):
    if len(list_player) != 4:
        print('Game chỉ cho phép có đúng 4 người chơi')
        return [-1,-1,-1,-1]
    
    env = np.full(62,0)
    count_win = [0,0,0,0]
    p_lst_idx = [0,1,2,3]
    for _n in range(num_game):
        rd.shuffle(p_lst_idx)
        file_per_2_new = [per_file_2[p_lst_idx[i]] for i in range(amount_player())]
        list_player_new = [list_player[p_lst_idx[i]] for i in range(amount_player())]
        winner, file_per_2_new = one_game_2(
            list_player_new, env,  file_per_2_new)

        list_p_id_new = [p_lst_idx.index(i) for i in range(amount_player())]
        per_file_2 = [file_per_2_new[list_p_id_new[i]] for i in range(amount_player())]
        
        count_win[p_lst_idx[winner]] += 1
    
    return count_win, per_file_2












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

@njit()
def data_to_layer_NhatAnh_130922(state,data0, data1):
    state = np.dot(state,data0)
    state *= state > 0
    active = state > 0
    layer1 = data1.flatten() * active
    return layer1

@njit()
def test2_NhatAnh_130922(state,file_per_2):
    layer = np.zeros(amount_action())
    for id in range(len(file_per_2[0])):
        layer += data_to_layer_NhatAnh_130922(state,file_per_2[0][id], file_per_2[1][id])
    base = np.zeros(amount_action())
    actions = get_list_action(state)
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
    actions = get_list_action(state)
    actions = np.where(actions == 1)[0]
    for act in base:
        if act in actions:
            return act
    ind = np.random.randint(len(actions))
    action = actions[ind]
    return action

@njit()
def test2_NhatAnh_200922(state,file_per_2):
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
    return np.random.choice(np.where(get_list_action(state) == 1)[0])

@njit()
def test2_NhatAnh_270922(state, file_per_2):
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
    a = get_list_action(play_state)
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
    a = get_list_action(play_state)
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

# @njit()
# def relu6_khanh_270922(x):
#     return np.minimum(np.maximum(0, x),6)

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
    a = get_list_action(play_state)
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

# list_activation_function = [Identity_an_130922, BinaryStep_an_130922, Sigmoid_an_130922, NegativePositiveStep_an_130922, Tanh_an_130922, ReLU_an_130922, LeakyReLU_an_130922, PReLU_an_130922, Gaussian_an_130922]
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
            # res_mat = list_activation_function[id](res_mat)
            res_mat = id_function_an_130922(id, res_mat, Identity_an_130922, BinaryStep_an_130922, Sigmoid_an_130922, NegativePositiveStep_an_130922, Tanh_an_130922, ReLU_an_130922, LeakyReLU_an_130922, PReLU_an_130922, Gaussian_an_130922)

    
    res_arr = res_mat[list_action]
    arr_max = np.where(res_arr == np.max(res_arr))[0]
    action_max_idx = np.random.choice(arr_max)
    return list_action[action_max_idx]

@njit()
def test2_An_130922(p_state, temp_file,  file_per_2):
    list_action = get_list_action(p_state)
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
        # res_mat = np.nan_to_num(res_mat)
        # res_mat = activation_function[data3i2](res_mat)
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
    list_action = get_list_action(p_state)
    list_action = np.where(list_action == 1)[0]
    if len(file_per_2) == 2: 
        type_file_per_2 = int(file_per_2[1][0][0][0])
    else:
        type_file_per_2 = int(file_per_2[0][1][0][0])
    
    if type_file_per_2 == 0: # fnn
        action = Ann_neural_network_an_200922(p_state, file_per_2[0], list_action)
    else: # sg
        # if len(file_per_2) < 3:
        #     file_per_2.append(file_per_2[0][0]/file_per_2[0][1])
            # print(file_per_2[0])
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
        # data3i = data[3*i]
        # data3i1 = data[3*i+1]
        # data3i2 = data[3*i+2]
        data3i = data[3*i]
        data3i1 = data[3*i+1].flatten()
        data3i2 = int(data[3*i+2][0][0])
        res_mat = np.dot(res_mat, data3i) + data3i1
        # res_mat = np.nan_to_num(res_mat)
        # res_mat = activation_function[data3i2](res_mat)
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
    list_action = get_list_action(p_state)
    list_action = np.where(list_action == 1)[0]
    if len(file_per_2) == 2: 
        type_file_per_2 = int(file_per_2[1][0][0][0])
    else:
        type_file_per_2 = int(file_per_2[0][1][0][0])
    if type_file_per_2 == 0:
        action = Ann_neural_network_an_270922(p_state, file_per_2[0], list_action)
        # return action, temp_file,  file_per_2
    else:
        # if len(file_per_2) < 3:
        #     file_per_2.append(file_per_2[0][0]/file_per_2[0][1])
        
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
    list_action = get_list_action(state)
    list_action = np.where(list_action == 1)[0]
    hidden1 = np.dot(state, file_per_2[0])
    hidden2 = hidden1 * (hidden1>0)
    values =  np.dot(hidden2, file_per_2[1])
    action = list_action[np.argmax(values[list_action])]
    return action

#########################################################
# @njit()
# def test2_Dat_200922(state,temp, file_per_2):
#     list_action = get_list_action(state)
#     list_action = np.where(list_action == 1)[0]
#     hidden1 = np.dot(state, file_per_2[0])
#     hidden2 = hidden1 * (hidden1>0)
#     values =  np.dot(hidden2, file_per_2[1])
#     action = list_action[np.argmax(values[list_action])]
#     return action,temp, file_per_2

# ################################################################
# @njit()
# def test2_Dat_270922(state,temp, file_per_2):
#     list_action = get_list_action(state)
#     list_action = np.where(list_action == 1)[0]
#     hidden1 = np.dot(state, file_per_2[0])
#     hidden2 = hidden1 * (hidden1>0)
#     values =  np.dot(hidden2, file_per_2[1])
#     action = list_action[np.argmax(values[list_action])]
#     return action,temp, file_per_2

###############################################################
###############################################################
###############################################################
@njit()
def neural_network_hieu_130922(state, file_temp0, file_temp1, file_temp2, list_action):
    norm_state = state/np.linalg.norm(state, 1)
    norm_state = np.tanh(norm_state)                    #dạng tanh
    norm_action = np.zeros(amount_action())
    norm_action[list_action] = 1
    norm_action = norm_action.reshape(1, amount_action())
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
    list_action = get_list_action(state)
    list_action = np.where(list_action == 1)[0]
    action = neural_network_hieu_130922(state, file_per_2[0], file_per_2[1], file_per_2[2], list_action)
    return action
#################################################################
@njit()
def agent_hieu_270922(state,file_temp,file_per):
    actions = get_list_action(state)
    actions = np.where(actions == 1)[0]
    action = np.random.choice(actions)
    file_per = (len(state),amount_action())
    return action,file_temp,file_per

# LEN_STATE_hieu_270922,AMOUNT_ACTION_hieu_270922 = normal_main([agent_hieu_270922]*amount_player(), 1, [0])[1]

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
    # norm_state = state.copy()
    norm_state = norm_state/np.linalg.norm(norm_state, 1)
    norm_state = softmax_hieu_270922(norm_state)
    norm_action = np.zeros(amount_action())
    norm_action[list_action] = 1
    norm_action = norm_action.reshape(1, amount_action())

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
    list_action = get_list_action(state)
    list_action = np.where(list_action == 1)[0]
    action = neural_network_hieu_270922(state, file_per_2[0], file_per_2[1], file_per_2[2], list_action)
    return action

######################################################################
######################################################################
######################################################################

@njit()
def file_temp_to_action_Phong_130922(state, file_temp):
    # print(state, file_temp)
    a = get_list_action(state)
    a = np.where(a == 1)[0]
    # print('action', a)
    RELU = np.ones(len(state))
    # print('hi')
    matrix_new = np.dot(RELU,file_temp)
    list_val_action = matrix_new[a]
    action = a[np.argmax(list_val_action)]
    # print('hhhhhhhh', action)
    return action

@njit() 
def test2_Phong_130922(state,file_per_2):
    action = file_temp_to_action_Phong_130922(state, file_per_2)
    # print(action)
    return action

#######################################################################

# @njit()
# def test2_Phong_200922(state,file_temp, file_per_2):
#     action = file_temp_to_action_Phong_130922(state, file_per_2)
#     return action

# ######################################################################

# @njit()
# def test2_Phong_270922(state,file_temp, file_per_2):
#     action = file_temp_to_action_Phong_130922(state, file_per_2)
#     return action













@njit()
def get_func(player_state, id, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11):
    if id == 0: return test2_An_270922(player_state, per0)
    elif id == 1: return test2_Dat_130922(player_state, per1)
    elif id == 2: return test2_Hieu_270922(player_state, per2)
    elif id == 3: return test2_Khanh_270922(player_state, per3)
    elif id == 4: return test2_NhatAnh_270922(player_state, per4)
    elif id == 5: return test2_Phong_130922(player_state, per5)
    elif id == 6: return test2_An_200922(player_state, per6)
    elif id == 7: return test2_Phong_130922(player_state, per7)
    elif id == 8: return test2_Dat_130922(player_state, per8)
    elif id == 9: return test2_Khanh_200922(player_state, per9)
    elif id == 10: return test2_Khanh_130922(player_state, per10)
    else: return test2_Dat_130922(player_state, per11)

@njit()
def one_game_numba(p0, list_other, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11):
    env = np.full(62,0)
    reset(env)
    while not check_env(env):
        reset(env)
    temp = [0]
    for i in range(1000):
        idx = env[52]
        player_state = get_player_state(env)
        if list_other[idx] == -1:
            
            action, temp, per_player = p0(player_state,temp,per_player)
        else:
            action = get_func(player_state, list_other[idx], per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11)
        arr_card_in_hand = step(action, env)
        if close_game(env) != -1:
            break
    
    for i in range(4):
        if list_other[idx] == -1:
            act, temp, per_player = p0(get_player_state(env), temp, per_player)
    winner = False
    if np.where(list_other == -1)[0] ==  close_game(env): winner = True
    else: winner = False
    return winner,  per_player


@njit()
def n_game_numba(p0, num_game, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11):
    win = 0
    for _n in range(num_game):
        list_other = np.append(np.random.choice(np.arange(12), 3), -1)
        np.random.shuffle(list_other)
        winner,per_player  = one_game_numba(p0, list_other, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11)
        win += winner
    return [win, num_game - win], per_player




def numba_main(p0, per_player, n_game):
    list_all_players = dict_game_for_player[game_name_]
    list_data = load_data_per2(list_all_players, game_name_)
    per0 = list_data[0]
    per1 = list_data[1]
    per2 = list_data[2]
    per3 = list_data[3]
    per4 = list_data[4]
    per5 = list_data[5]
    per6 = list_data[6]
    per7 = list_data[7]
    per8 = list_data[8]
    per9 = list_data[9]
    per10 = list_data[10]
    per11 = list_data[11]
    return n_game_numba(p0, n_game, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11)