from numba import njit
import numpy as np
import random as rd
# from colorama import Fore, Style

all_action = np.array([[0,0],[1,0],[1,1],[1,2],[1,3],[1,4],[1,5],[1,6],[1,7],[1,8],[1,9],[1,10],[1,11],[1,12],[1,13],[1,14],[1,15],[1,16],[1,17],[1,18],[1,19],[1,20],[1,21],[1,22],[1,23],[1,24],[1,25],[1,26],[1,27],[1,28],[1,29],[1,30],[1,31],[1,32],[1,33],[1,34],[1,35],[1,36],[1,37],[1,38],[1,39],[1,40],[1,41],[1,42],[1,43],[1,44],[1,45],[1,46],[1,47],[1,48],[1,49],[1,50],[1,51],[2,1],[2,2],[2,3],[2,5],[2,6],[2,7],[2,9],[2,10],[2,11],[2,13],[2,14],[2,15],[2,17],[2,18],[2,19],[2,21],[2,22],[2,23],[2,25],[2,26],[2,27],[2,29],[2,30],[2,31],[2,33],[2,34],[2,35],[2,37],[2,38],[2,39],[2,41],[2,42],[2,43],[2,45],[2,46],[2,47],[2,49],[2,50],[2,51],[3,2],[3,3],[3,6],[3,7],[3,10],[3,11],[3,14],[3,15],[3,18],[3,19],[3,22],[3,23],[3,26],[3,27],[3,30],[3,31],[3,34],[3,35],[3,38],[3,39],[3,42],[3,43],[3,46],[3,47],[3,50],[3,51],[4,3],[4,7],[4,11],[4,15],[4,19],[4,23],[4,27],[4,31],[4,35],[4,39],[4,43],[4,47],[5,8],[5,9],[5,10],[5,11],[5,12],[5,13],[5,14],[5,15],[5,16],[5,17],[5,18],[5,19],[5,20],[5,21],[5,22],[5,23],[5,24],[5,25],[5,26],[5,27],[5,28],[5,29],[5,30],[5,31],[5,32],[5,33],[5,34],[5,35],[5,36],[5,37],[5,38],[5,39],[5,40],[5,41],[5,42],[5,43],[5,44],[5,45],[5,46],[5,47],[6,12],[6,13],[6,14],[6,15],[6,16],[6,17],[6,18],[6,19],[6,20],[6,21],[6,22],[6,23],[6,24],[6,25],[6,26],[6,27],[6,28],[6,29],[6,30],[6,31],[6,32],[6,33],[6,34],[6,35],[6,36],[6,37],[6,38],[6,39],[6,40],[6,41],[6,42],[6,43],[6,44],[6,45],[6,46],[6,47],[7,16],[7,17],[7,18],[7,19],[7,20],[7,21],[7,22],[7,23],[7,24],[7,25],[7,26],[7,27],[7,28],[7,29],[7,30],[7,31],[7,32],[7,33],[7,34],[7,35],[7,36],[7,37],[7,38],[7,39],[7,40],[7,41],[7,42],[7,43],[7,44],[7,45],[7,46],[7,47],[8,20],[8,21],[8,22],[8,23],[8,24],[8,25],[8,26],[8,27],[8,28],[8,29],[8,30],[8,31],[8,32],[8,33],[8,34],[8,35],[8,36],[8,37],[8,38],[8,39],[8,40],[8,41],[8,42],[8,43],[8,44],[8,45],[8,46],[8,47],[9,24],[9,25],[9,26],[9,27],[9,28],[9,29],[9,30],[9,31],[9,32],[9,33],[9,34],[9,35],[9,36],[9,37],[9,38],[9,39],[9,40],[9,41],[9,42],[9,43],[9,44],[9,45],[9,46],[9,47],[10,28],[10,29],[10,30],[10,31],[10,32],[10,33],[10,34],[10,35],[10,36],[10,37],[10,38],[10,39],[10,40],[10,41],[10,42],[10,43],[10,44],[10,45],[10,46],[10,47],[11,32],[11,33],[11,34],[11,35],[11,36],[11,37],[11,38],[11,39],[11,40],[11,41],[11,42],[11,43],[11,44],[11,45],[11,46],[11,47],[12,36],[12,37],[12,38],[12,39],[12,40],[12,41],[12,42],[12,43],[12,44],[12,45],[12,46],[12,47],[13,40],[13,41],[13,42],[13,43],[13,44],[13,45],[13,46],[13,47],[14,9],[14,10],[14,11],[14,13],[14,14],[14,15],[14,17],[14,18],[14,19],[14,21],[14,22],[14,23],[14,25],[14,26],[14,27],[14,29],[14,30],[14,31],[14,33],[14,34],[14,35],[14,37],[14,38],[14,39],[14,41],[14,42],[14,43],[14,45],[14,46],[14,47],[15,13],[15,14],[15,15],[15,17],[15,18],[15,19],[15,21],[15,22],[15,23],[15,25],[15,26],[15,27],[15,29],[15,30],[15,31],[15,33],[15,34],[15,35],[15,37],[15,38],[15,39],[15,41],[15,42],[15,43],[15,45],[15,46],[15,47]])

# -------------------- NOPYTHON FUNCTIONS --------------------
@njit
def initEnv(e_state):
    temp = np.arange(52)
    np.random.shuffle(temp)
    for i in range(4):
        e_state[temp[13*i:13*(i+1)]] = i

    e_state[52] = 0
    e_state[53:57] = 1
    e_state[57] = 0
    e_state[58:60] = 0

@njit
def getAgentState(e_state):
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
def getValidActions(player_state_origin:np.int64):
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
def checkEnded(e_state):
    for i in range(4):
        if np.count_nonzero(e_state[0:52] == i) == 0:
            return i
    
    return -1

@njit
def getActionSize():
    return 403

@njit
def getAgentSize():
    return 4

@njit()
def getStateSize():
    return 60

@njit
def getReward(p_state):
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
def stepEnv(action, e_state):
    # p_state = getAgentState(e_state)
    # arr_action = getValidActions(p_state)

    # if arr_action[action] != 1:
    #     '''
    #     Action không hợp lệ
    #     '''
    #     print('Action không hợp lệ')
    # else:
    hand = all_action[action]
    # p_state[np.where(e_state[0:52] == e_state[52])[0]] = 0 # Dòng này lấy ra từ get_player_state
    # arr_card = np.where(p_state[0:52] == 0)[0]
    arr_card = np.where(e_state[0:52] == e_state[52])[0]  # Câu lệnh này cho ra kết quả tương đương với câu lệnh liền trên
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
    initEnv(env)
    while not check_env(env):
        initEnv(env)
    
    temp_file = [[0], [0], [0], [0]]
    while True:
        p_state = getAgentState(env)
        list_action = getValidActions(p_state)
        act, per_file = list_player[env[52]](p_state, per_file)
        if list_action[act] != 1:
            raise Exception('Action không hợp lệ')
        arr_card_in_hand = stepEnv(act, env)
        if checkEnded(env) != -1:
            break
    
    winner = checkEnded(env)
    for i in range(4):
        env[52] = i
        act, per_file = list_player[env[52]](getAgentState(env), per_file)
    
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
    initEnv(env)
    while not check_env(env):
        initEnv(env)



    while True:
        p_idx = env[52]
        p_state = getAgentState(env)
        list_action = getValidActions(p_state)
        if p_lst_idx_shuffle[p_idx] == 0:
            act, per_file = p0(p_state, per_file)
        elif p_lst_idx_shuffle[p_idx] == 1:
            act, per_file = p1(p_state, per_file)
        elif p_lst_idx_shuffle[p_idx] == 2:
            act, per_file = p2(p_state, per_file)
        else:
            act, per_file = p3(p_state, per_file)

        if list_action[act] != 1:
            raise Exception('Action không hợp lệ')
        arr_card_in_hand = stepEnv(act, env)
        if checkEnded(env) != -1:
            break

    
    winner = checkEnded(env)
    for p_idx in range(4):
        env[52] = p_idx
        p_state = getAgentState(env)
        if p_lst_idx_shuffle[p_idx] == 0:
            act, per_file = p0(p_state, per_file)
        elif p_lst_idx_shuffle[p_idx] == 1:
            act, per_file = p1(p_state, per_file)
        elif p_lst_idx_shuffle[p_idx] == 2:
            act, per_file = p2(p_state, per_file)
        else:
            act, per_file = p3(p_state, per_file)
    
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


from numba.typed import List
from base.TLMN.AgentEnv import *


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
    initEnv(env)
    while not check_env(env):
        initEnv(env)
    while True:
        idx = env[52]
        player_state = getAgentState(env)
        list_action = getValidActions(player_state)
        if list_other[idx] == -1:   
            action, per_player = p0(player_state,per_player)
        elif list_other[idx] == -2:
            action = random_Env(player_state)
        else:
            action = get_func(player_state, list_other[idx], per0, per1, per2, per3, per4, per5, per6, per7, per8)
        
        if list_action[action] != 1:
            raise Exception('Action không hợp lệ')
        arr_card_in_hand = stepEnv(action, env)
        if checkEnded(env) != -1:
            break
    
    for i in range(4):
        env[52] = i
        if list_other[i] == -1:
            act, per_player = p0(getAgentState(env), per_player)
    winner = False
    if np.where(list_other == -1)[0] ==  checkEnded(env): winner = True
    else: winner = False
    return winner,  per_player


@njit()
def n_game_numba(p0, num_game, per_player, level, per0, per1, per2, per3, per4, per5, per6, per7, per8):
    win = 0
    if level == 0:
        list_other = np.array([-2, -2, -2, -1])
    elif level == 1:
        list_other = np.array([6, 4, 2, -1])
    elif level == 2:
        list_other = np.array([7, 1, 8, -1])

    else:
        raise Exception('Hiện tại không có level này')
    for _n in range(num_game):
        np.random.shuffle(list_other)
        winner,per_player  = one_game_numba(p0, list_other, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8)
        win += winner
    return win, per_player


def numba_main_2(p0, n_game, per_player, level):
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
    return n_game_numba(p0, n_game, per_player, level, per0, per1, per2, per3, per4, per5, per6, per7, per8)


# @njit()
def one_game_numba_2(p0, list_other, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8):
    env = np.full(60,0)
    initEnv(env)
    while not check_env(env):
        initEnv(env)
    while True:
        idx = env[52]
        player_state = getAgentState(env)
        list_action = getValidActions(player_state)
        if list_other[idx] == -1:   
            action, per_player = p0(player_state,per_player)
        elif list_other[idx] == -2:
            action = random_Env(player_state)
        else:
            action = get_func(player_state, list_other[idx], per0, per1, per2, per3, per4, per5, per6, per7, per8)
        
        if list_action[action] != 1:
            raise Exception('Action không hợp lệ')
        arr_card_in_hand = stepEnv(action, env)
        if checkEnded(env) != -1:
            break
    
    for i in range(4):
        env[52] = i
        if list_other[i] == -1:
            act, per_player = p0(getAgentState(env), per_player)
    winner = False
    if np.where(list_other == -1)[0] ==  checkEnded(env): winner = True
    else: winner = False
    return winner,  per_player


# @njit()
def n_game_numba_2(p0, num_game, per_player, level, per0, per1, per2, per3, per4, per5, per6, per7, per8):
    win = 0
    if level == 0:
        list_other = np.array([-2, -2, -2, -1])
    elif level == 1:
        list_other = np.array([6, 4, 2, -1])
    elif level == 2:
        list_other = np.array([7, 1, 8, -1])

    else:
        raise Exception('Hiện tại không có level này')
    for _n in range(num_game):
        np.random.shuffle(list_other)
        winner,per_player  = one_game_numba_2(p0, list_other, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8)
        win += winner
    return win, per_player


def normal_main_2(p0, n_game, per_player, level):
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
    return n_game_numba_2(p0, n_game, per_player, level, per0, per1, per2, per3, per4, per5, per6, per7, per8)

