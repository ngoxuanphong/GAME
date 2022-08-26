from numba import njit
import numpy as np
import random as rd
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

    return p_state

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
def get_list_action(p_state):
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
    if action not in list_action:
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