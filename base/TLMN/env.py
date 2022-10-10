from numba import njit
import numpy as np
import random as rd
# from colorama import Fore, Style

all_action = np.array([[0,0],[1,0],[1,1],[1,2],[1,3],[1,4],[1,5],[1,6],[1,7],[1,8],[1,9],[1,10],[1,11],[1,12],[1,13],[1,14],[1,15],[1,16],[1,17],[1,18],[1,19],[1,20],[1,21],[1,22],[1,23],[1,24],[1,25],[1,26],[1,27],[1,28],[1,29],[1,30],[1,31],[1,32],[1,33],[1,34],[1,35],[1,36],[1,37],[1,38],[1,39],[1,40],[1,41],[1,42],[1,43],[1,44],[1,45],[1,46],[1,47],[1,48],[1,49],[1,50],[1,51],[2,1],[2,2],[2,3],[2,5],[2,6],[2,7],[2,9],[2,10],[2,11],[2,13],[2,14],[2,15],[2,17],[2,18],[2,19],[2,21],[2,22],[2,23],[2,25],[2,26],[2,27],[2,29],[2,30],[2,31],[2,33],[2,34],[2,35],[2,37],[2,38],[2,39],[2,41],[2,42],[2,43],[2,45],[2,46],[2,47],[2,49],[2,50],[2,51],[3,2],[3,3],[3,6],[3,7],[3,10],[3,11],[3,14],[3,15],[3,18],[3,19],[3,22],[3,23],[3,26],[3,27],[3,30],[3,31],[3,34],[3,35],[3,38],[3,39],[3,42],[3,43],[3,46],[3,47],[3,50],[3,51],[4,3],[4,7],[4,11],[4,15],[4,19],[4,23],[4,27],[4,31],[4,35],[4,39],[4,43],[4,47],[5,8],[5,9],[5,10],[5,11],[5,12],[5,13],[5,14],[5,15],[5,16],[5,17],[5,18],[5,19],[5,20],[5,21],[5,22],[5,23],[5,24],[5,25],[5,26],[5,27],[5,28],[5,29],[5,30],[5,31],[5,32],[5,33],[5,34],[5,35],[5,36],[5,37],[5,38],[5,39],[5,40],[5,41],[5,42],[5,43],[5,44],[5,45],[5,46],[5,47],[6,12],[6,13],[6,14],[6,15],[6,16],[6,17],[6,18],[6,19],[6,20],[6,21],[6,22],[6,23],[6,24],[6,25],[6,26],[6,27],[6,28],[6,29],[6,30],[6,31],[6,32],[6,33],[6,34],[6,35],[6,36],[6,37],[6,38],[6,39],[6,40],[6,41],[6,42],[6,43],[6,44],[6,45],[6,46],[6,47],[7,16],[7,17],[7,18],[7,19],[7,20],[7,21],[7,22],[7,23],[7,24],[7,25],[7,26],[7,27],[7,28],[7,29],[7,30],[7,31],[7,32],[7,33],[7,34],[7,35],[7,36],[7,37],[7,38],[7,39],[7,40],[7,41],[7,42],[7,43],[7,44],[7,45],[7,46],[7,47],[8,20],[8,21],[8,22],[8,23],[8,24],[8,25],[8,26],[8,27],[8,28],[8,29],[8,30],[8,31],[8,32],[8,33],[8,34],[8,35],[8,36],[8,37],[8,38],[8,39],[8,40],[8,41],[8,42],[8,43],[8,44],[8,45],[8,46],[8,47],[9,24],[9,25],[9,26],[9,27],[9,28],[9,29],[9,30],[9,31],[9,32],[9,33],[9,34],[9,35],[9,36],[9,37],[9,38],[9,39],[9,40],[9,41],[9,42],[9,43],[9,44],[9,45],[9,46],[9,47],[10,28],[10,29],[10,30],[10,31],[10,32],[10,33],[10,34],[10,35],[10,36],[10,37],[10,38],[10,39],[10,40],[10,41],[10,42],[10,43],[10,44],[10,45],[10,46],[10,47],[11,32],[11,33],[11,34],[11,35],[11,36],[11,37],[11,38],[11,39],[11,40],[11,41],[11,42],[11,43],[11,44],[11,45],[11,46],[11,47],[12,36],[12,37],[12,38],[12,39],[12,40],[12,41],[12,42],[12,43],[12,44],[12,45],[12,46],[12,47],[13,40],[13,41],[13,42],[13,43],[13,44],[13,45],[13,46],[13,47],[14,9],[14,10],[14,11],[14,13],[14,14],[14,15],[14,17],[14,18],[14,19],[14,21],[14,22],[14,23],[14,25],[14,26],[14,27],[14,29],[14,30],[14,31],[14,33],[14,34],[14,35],[14,37],[14,38],[14,39],[14,41],[14,42],[14,43],[14,45],[14,46],[14,47],[15,13],[15,14],[15,15],[15,17],[15,18],[15,19],[15,21],[15,22],[15,23],[15,25],[15,26],[15,27],[15,29],[15,30],[15,31],[15,33],[15,34],[15,35],[15,37],[15,38],[15,39],[15,41],[15,42],[15,43],[15,45],[15,46],[15,47]])

# -------------------- NOPYTHON FUNCTIONS --------------------
@njit
def reset(e_state):
    temp = np.arange(52)
    np.random.shuffle(temp)
    for i in range(4):
        e_state[temp[13*i:13*(i+1)]] = i

    e_state[52] = 0
    e_state[53:57] = 1
    e_state[57] = 0
    e_state[58:60] = 0

@njit
def get_player_state(e_state):
    p_state = np.full(60,0)

    p_state[0:52] = e_state[0:52]
    p_state[np.where(e_state[0:52] == e_state[52])[0]] = 0
    p_state[np.where((e_state[0:52] != e_state[52]) & (e_state[0:52] != -1))[0]] = 1

    temp = np.arange(4)
    p_state[52:55] = e_state[53+np.concatenate((temp[e_state[52]+1:4], temp[0:e_state[52]]), axis=0)]

    for i in range(1,4):
        p_state[54+i] = np.count_nonzero(e_state[0:52] == (e_state[52]+i)%4)
    
    if e_state[52] == e_state[57]:
        p_state[58:60] = 0
    else:
        p_state[58:60] = e_state[58:60]
    
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

    arr_action = []
    for hand in possible_hands:
        if hand[0] == 0:
            arr_action.append(0)
        elif hand[0] >= 1 and hand[0] <= 4:
            arr_action.append(13*np.sum(np.arange(4,5-hand[0],-1)) + hand[1] - (hand[0]-1)*(hand[1]//4) - hand[0] + 2)
        elif hand[0] >= 5 and hand[0] <= 13:
            arr_action.append(4*np.sum(np.arange(10,15-hand[0],-1)) + hand[1] - 4*hand[0] + 142)
        else:
            arr_action.append(27*hand[0] + hand[1] - (hand[1]//4) - 39)
    
    return np.array(arr_action)

@njit
def get_list_action(player_state_origin:np.int64):
    list_action_return = np.zeros(403)
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

    # arr_action = []
    for hand in possible_hands:
        if hand[0] == 0:
            list_action_return[0] = 1
        elif hand[0] >= 1 and hand[0] <= 4:
            list_action_return[13*np.sum(np.arange(4,5-hand[0],-1)) + hand[1] - (hand[0]-1)*(hand[1]//4) - hand[0] + 2] = 1
        elif hand[0] >= 5 and hand[0] <= 13:
            list_action_return[4*np.sum(np.arange(10,15-hand[0],-1)) + hand[1] - 4*hand[0] + 142] = 1
        else:
            list_action_return[27*hand[0] + hand[1] - (hand[1]//4) - 39] = 1

    return list_action_return


@njit
def close_game(e_state):
    for i in range(4):
        if np.count_nonzero(e_state[0:52] == i) == 0:
            return i
    
    return -1

@njit
def amount_action():
    return 403

@njit
def amount_player():
    return 4

@njit()
def amount_state():
    return 60

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
    arr_action = get_list_action(p_state)

    if arr_action[action] != 1:
        '''
        Action không hợp lệ
        '''
        print('Action không hợp lệ')
    else:
        hand = all_action[action]
        arr_card = np.where(p_state[0:52] == 0)[0]
        arr_card_in_hand = []
        if hand[0] == 0:
            pass
        elif hand[0] >= 1 and hand[0] <= 4:
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

    if hand[0] == 0:
        e_state[53:57][e_state[52]] = 0
    else:
        e_state[57] = e_state[52]
        e_state[58:60] = hand

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
#     print(Fore.LIGHTCYAN_EX + 'Chọn action' + Style.RESET_ALL, act, all_action[act])
#     print(Fore.LIGHTCYAN_EX + 'Các lá bài trong action:' + Style.RESET_ALL, end=' ')
#     print_list_card(convert(arr_card_in_hand))
#     print('----------------------------------------------------------------------------------------------------')
#     print_player_cards(e_state)


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


def normal_main(list_player, num_game, per_file):
    if len(list_player) != 4:
        print('Game chỉ cho phép có đúng 4 người chơi')
        return [-1,-1,-1,-1], per_file
    
    env = np.full(60,0)
    count_win = [0,0,0,0]
    p_lst_idx = [0,1,2,3]
    for _n in range(num_game):
        rd.shuffle(p_lst_idx)
        winner, per_file = one_game(
            [list_player[p_lst_idx[0]], list_player[p_lst_idx[1]], list_player[p_lst_idx[2]], list_player[p_lst_idx[3]]], env, per_file
        )

        count_win[p_lst_idx[winner]] += 1
    
    return count_win, per_file


@njit()
def numba_one_game(p_lst_idx_shuffle, p0, p1, p2, p3, env, per_file):
    reset(env)
    while not check_env(env):
        reset(env)

    temp_1_player = List()
    temp_1_player.append(np.array([[0.]]))
    temp_file = [temp_1_player]*(amount_player())

    while True:
        p_idx = env[52]
        p_state = get_player_state(env)
        if p_lst_idx_shuffle[p_idx] == 0:
            act, temp_file[p_idx], per_file = p0(p_state, temp_file[p_idx], per_file)
        elif p_lst_idx_shuffle[p_idx] == 1:
            act, temp_file[p_idx], per_file = p1(p_state, temp_file[p_idx], per_file)
        elif p_lst_idx_shuffle[p_idx] == 2:
            act, temp_file[p_idx], per_file = p2(p_state, temp_file[p_idx], per_file)
        else:
            act, temp_file[p_idx], per_file = p3(p_state, temp_file[p_idx], per_file)

        arr_card_in_hand = step(act, env)
        if close_game(env) != -1:
            break

    
    winner = close_game(env)
    for p_idx in range(4):
        env[52] = p_idx
        p_state = get_player_state(env)
        if p_lst_idx_shuffle[p_idx] == 0:
            act, temp_file[p_idx], per_file = p0(p_state, temp_file[p_idx], per_file)
        elif p_lst_idx_shuffle[p_idx] == 1:
            act, temp_file[p_idx], per_file = p1(p_state, temp_file[p_idx], per_file)
        elif p_lst_idx_shuffle[p_idx] == 2:
            act, temp_file[p_idx], per_file = p2(p_state, temp_file[p_idx], per_file)
        else:
            act, temp_file[p_idx], per_file = p3(p_state, temp_file[p_idx], per_file)
    
    return winner, per_file

@njit()
def numba_main(p0, p1, p2, p3, num_game,per_file):
    env = np.full(60,0)
    count_win = [0,0,0,0]
    p_lst_idx = np.array([0,1,2,3])
    for _n in range(num_game):
        np.random.shuffle(p_lst_idx)
        winner, per_file = numba_one_game(p_lst_idx, p0, p1, p2, p3, env, per_file )
        count_win[p_lst_idx[winner]] += 1

    return count_win, per_file

















from system.mainFunc import dict_game_for_player, load_data_per2  
game_name_ = 'TLMN'
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
    list_action = get_list_action(p_state)
    list_action = np.where(list_action == 1)[0]
    if len(file_per_2) == 2: 
        type_file_per_2 = int(file_per_2[1][0][0][0])
    else:
        type_file_per_2 = int(file_per_2[0][1][0][0])
    if type_file_per_2 == 0:
        action = Ann_neural_network_an_270922(p_state, file_per_2[0], list_action)
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
    list_action = get_list_action(state)
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
    a = get_list_action(state)
    a = np.where(a == 1)[0]
    RELU = np.ones(len(state))
    matrix_new = np.dot(RELU,file_temp)
    list_val_action = matrix_new[a]
    action = a[np.argmax(list_val_action)]
    return action

@njit() 
def test2_Phong_130922(state,file_per_2):
    action = file_temp_to_action_Phong_130922(state, file_per_2)
    return action

#######################################################################














@njit()
def get_func(player_state, id, per0, per1, per2, per3, per4, per5, per6, per7, per8):
    if id == 0: return test2_An_270922(player_state, per0)
    elif id == 1: return test2_Dat_130922(player_state, per1)
    elif id == 2: return test2_Khanh_270922(player_state, per2)
    elif id == 3: return test2_An_200922(player_state, per3)
    elif id == 4: return test2_Phong_130922(player_state, per4)
    elif id == 5: return test2_Dat_130922(player_state, per5)
    elif id == 6: return test2_Khanh_200922(player_state, per6)
    elif id == 7: return test2_Khanh_130922(player_state, per7)
    else: return test2_Dat_130922(player_state, per8)

@njit()
def one_game_numba(p0, list_other, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8):
    env = np.full(60,0)
    reset(env)
    while not check_env(env):
        reset(env)
    _temp_ = List()
    _temp_.append(np.array([[0]]))
    while True:
        idx = env[52]
        player_state = get_player_state(env)
        if list_other[idx] == -1:
            
            action, _temp_, per_player = p0(player_state,_temp_,per_player)
        else:
            action = get_func(player_state, list_other[idx], per0, per1, per2, per3, per4, per5, per6, per7, per8)
        arr_card_in_hand = step(action, env)
        if close_game(env) != -1:
            break
    
    for i in range(4):
        env[52] = i
        if list_other[i] == -1:
            act, _temp_, per_player = p0(get_player_state(env), _temp_, per_player)
    winner = False
    if np.where(list_other == -1)[0] ==  close_game(env): winner = True
    else: winner = False
    return winner,  per_player


@njit()
def n_game_numba(p0, num_game, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8):
    win = 0
    for _n in range(num_game):
        list_other = np.append(np.random.choice(np.arange(9), 3), -1)
        np.random.shuffle(list_other)
        winner,per_player  = one_game_numba(p0, list_other, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8)
        win += winner
    return win, per_player


def numba_main_2(p0, per_player, n_game):
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
    return n_game_numba(p0, n_game, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8)

# @njit()
def one_game_numba_2(p0, list_other, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8):
    env = np.full(60,0)
    reset(env)
    while not check_env(env):
        reset(env)
    _temp_ = List()
    _temp_.append(np.array([[0]]))
    while True:
        idx = env[52]
        player_state = get_player_state(env)
        if list_other[idx] == -1:
            
            action, _temp_, per_player = p0(player_state,_temp_,per_player)
        else:
            action = get_func(player_state, list_other[idx], per0, per1, per2, per3, per4, per5, per6, per7, per8)
        arr_card_in_hand = step(action, env)
        if close_game(env) != -1:
            break
    
    for i in range(4):
        env[52] = i
        if list_other[i] == -1:
            act, _temp_, per_player = p0(get_player_state(env), _temp_, per_player)
    winner = False
    if np.where(list_other == -1)[0] ==  close_game(env): winner = True
    else: winner = False
    return winner,  per_player


# @njit()
def n_game_numba_2(p0, num_game, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8):
    win = 0
    for _n in range(num_game):
        list_other = np.append(np.random.choice(np.arange(9), 3), -1)
        np.random.shuffle(list_other)
        winner,per_player  = one_game_numba_2(p0, list_other, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8)
        win += winner
    return win, per_player


def normal_main_2(p0, n_game):
    per_player = 0
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
    return n_game_numba_2(p0, n_game, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8)

