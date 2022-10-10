from operator import index
import numpy as np
import random
import multiprocessing as mp
import time
from numba import jit, njit

@njit
def reset(n):
    '''
    n : số lượng người chơi\n
    id| Name card    | Amount\n
    0 | Tempura      | 14\n
    1 | Sashimi      | 14\n
    2 | Dumpling     | 14\n
    3 | 1 Maki Roll  | 6\n
    4 | 2 Maki Roll  | 12\n
    5 | 3 Maki Roll  | 8\n
    6 | Salmon Nigiri| 10\n
    7 | Squid Nigiri | 5\n
    8 | Egg Nigiri   | 5\n
    9| Pudding      | 10\n
    10| Wasabi       | 6\n
    11| Chopsticks   | 4\n
    '''
    amount_card = np.array([14,14,14,6,12,8,10,5,5,10,6,4])
    id_card = np.array([0,1,2,3,4,5,6,7,8,9,10,11])
    state_sys = np.array([1,1,n])
    list_card = np.array([0])
    index = 0
    for amount in amount_card:
        list_c = np.array([id_card[index] for i in range(amount)])
        if len(list_card) == 1:
            list_card = list_c
        else:
            list_card = np.concatenate((list_card,list_c))
        index += 1
    np.random.shuffle(list_card)
    cards = list_card[:3*n*(12-n)]
    state_sys = np.concatenate((state_sys,cards))
    for i in range(n):
        state_sys =  np.concatenate((state_sys ,np.array([0,0]) ,np.array([-1 for i in range(12-n)])))
    return state_sys

@njit
def get_player_state(state_sys,index_player):
    amount_player = state_sys[2]
    round = state_sys[0] - 1
    turn = state_sys[1]+4
    index_start_card_board = (((turn-index_player) % amount_player))*(12-amount_player) + round*(12-amount_player)*amount_player + 3
    # index_end_card_board = (((index_player+turn) % amount_player)+1)*(12-amount_player) + round*(12-amount_player)*amount_player + 3   
    index_end_card_board =  index_start_card_board + (12-amount_player)
    state_player = np.array([0])
    len_action = 1
    for index_player_relative in range(index_player, index_player + amount_player):
        index_start_player_s = (index_player_relative%amount_player) * (12-amount_player) + 3*amount_player*(12-amount_player) + (index_player_relative % amount_player+1) *2 +1
        index_end_player_s = (index_player_relative % amount_player+1) * (12-amount_player) + 3*amount_player*(12-amount_player) + (index_player_relative % amount_player+2)*2 +1
        if index_player == index_player_relative:
            state_player = np.concatenate((state_sys[:3],state_sys[index_start_card_board:index_end_card_board],state_sys[index_start_player_s:index_end_player_s]))
        else:
            state_player = np.concatenate((state_player,state_sys[index_start_player_s:index_end_player_s]))
    return np.append(state_player,np.array([0,len_action])).astype(np.float64)

@njit
def caculater_for_one(card):
    score_dumpling = [0,1,3,6,10,15]
    score = card[0]
    card = card[2:]
    # print("card",card,end = " ")
    tempura = np.count_nonzero(card == 0)
   
    score += tempura//2 * 5
    # print("tempura: ",score,end = " ")

    sashimi = np.count_nonzero(card == 1)
    score += sashimi//3 * 10
    # print("sashimi: ",score,end = " ")

    dumpling = np.count_nonzero(card ==2)
    if dumpling>5:
        score += score_dumpling[dumpling-5]
        dumpling = 5
    score += score_dumpling[dumpling]

    # print("dumpling: ",score,end = " ")

    salmon_nigiri = np.count_nonzero(card ==6)
    squid_nigiri = np.count_nonzero(card ==7)
    egg_nigiri = np.count_nonzero(card ==8)
    wasabi = np.count_nonzero(card ==10)
    list_ = [squid_nigiri,salmon_nigiri,egg_nigiri]
    for i in range(len(list_)):
        for j in range(list_[i]):
            if wasabi != 0:
                score += (3-i)*3
                wasabi -= 1
            else:
                score += (3-i)

    return score,np.count_nonzero(card ==9)

@njit
def count_maki(card):
    maki_1 = np.count_nonzero(card == 3)
    maki_2 = np.count_nonzero(card == 4)
    maki_3 = np.count_nonzero(card == 5)
    return maki_1 + maki_2*2 + maki_3*3

@njit
def get_index(arr,first,second):
    return np.where(arr == first)[0],np.where(arr == second)[0]

@njit
def caculator_pudding(state_sys,amount_player):
    amount_player = state_sys[2]
    arr_pudding = np.array([0 for i in range(amount_player)])
    for index_player_relative in range(0, amount_player):
        index_start_player_s = (index_player_relative) * (12-amount_player) + 3*amount_player*(12-amount_player) + (index_player_relative+1) *2 +1
        pudding = state_sys[index_start_player_s+1]
        arr_pudding[index_player_relative] = pudding

    max_p, min_p = max(arr_pudding),min(arr_pudding)
    list_ = get_index(arr_pudding,max_p,min_p)
    # print(list_)
    for top in range(len(list_)):
        for index_player_relative in list_[top]:
            score = (6 - top*12) // len(list_[top])
            index_start_player_s = (index_player_relative) * (12-amount_player) + 3*amount_player*(12-amount_player) + (index_player_relative+1) *2 +1
            state_sys[index_start_player_s] += score
    return state_sys

@njit
def caculater_score(state_sys,amount_player):
    amount_player = state_sys[2]
    round = state_sys[0] - 1
    arr_maki = np.array([0 for i in range(amount_player)])
    first,second = -1,-1
    if round != 4:
        for index_player_relative in range(0, amount_player):
            index_start_player_s = (index_player_relative) * (12-amount_player) + 3*amount_player*(12-amount_player) + (index_player_relative+1) *2 +1
            index_end_player_s = (index_player_relative+1) * (12-amount_player) + 3*amount_player*(12-amount_player) + (index_player_relative+2)*2 +1
            card = state_sys[index_start_player_s:index_end_player_s]
            state_sys[index_start_player_s],puding = caculater_for_one(card)
            state_sys[index_start_player_s+1] += puding
            c_maki = count_maki(card)
            arr_maki[index_player_relative] = c_maki
            if c_maki > first:
                second = first
                first = c_maki
            elif c_maki > second:
                second = c_maki

        list_ = get_index(arr_maki,first,second)
        # print(list_)
        for top in range(len(list_)):
            for index_player_relative in list_[top]:

                score = (6 - top*3) // len(list_[top])
                index_start_player_s = (index_player_relative) * (12-amount_player) + 3*amount_player*(12-amount_player) + (index_player_relative+1) *2 +1
                state_sys[index_start_player_s] = state_sys[index_start_player_s] + score
                # print(score)
            if len(list_[top]) > 1:
                break
    return state_sys

@njit
def check_victory(state_player):
    if state_player[1] <= (12-state_player[2])*3:
        return -1
    amount_player = int(state_player[2])
    list_score = state_player[15-amount_player::14-amount_player]
    Max_Score = max(list_score)
    list_winner = np.where(list_score==Max_Score)[0]
    check_winer_self = np.where(list_winner==0)[0]
    if len(check_winer_self) == 1:
        if len(list_winner) == 1:
            return 1
        else:
            amount_player = int(state_player[2])
            list_pudding = state_player[15-amount_player+1::14-amount_player]
            pudding_victoryer = list_pudding[list_winner]
            max_puding = max(pudding_victoryer)
            if list_pudding[0] == max_puding:
                return 1
            else:
                return 0
    else:
        return 0

@njit
def winner_victory(state_sys):
    amount_player = int(state_sys[2])
    list_score = state_sys[3+3*amount_player*(12-amount_player)::14-amount_player]
    max_score = max(list_score)
    winner = np.where(list_score == max_score)[0]
    # print("Diem:",list_score)
    if len(winner) == 1:
        return winner
    else:
        list_pudding = state_sys[3+3*amount_player*(12-amount_player)+1::14-amount_player]
        pudding_victoryer = list_pudding[winner]
        max_puding = max(pudding_victoryer)
        winner_puding = np.where(pudding_victoryer == max_puding)[0]
        return winner[winner_puding]


@njit
def step(state_sys,list_action,amount_player,turn,round):
    player = 0
    turn +=4
    for a in range(turn+amount_player,turn,-1):
        index_board_s = (a % amount_player)*(12-amount_player) + round*(12-amount_player)*amount_player + 3
        index_board_e = index_board_s + (12 - amount_player)
        index_player_s = (player % amount_player) * (12 - amount_player) + 3*amount_player*(12-amount_player) + (player % amount_player+1) *2 +3
        index_player_e = index_player_s+ (12 - amount_player)
        # print(index_player_s,index_player_e,index_board_s,index_board_e)
        l_a = list_action[player]
        l_a = l_a[np.where(l_a >= 0)[0]]
        for i in l_a:
            if i == 13:
                break
            if i == 12:
                state_sys = move_card(state_sys,11,amount_player,index_player_s,index_player_e,index_board_s,index_board_e)
                continue
            state_sys = move_card(state_sys,i,amount_player,index_board_s,index_board_e,index_player_s,index_player_e)
        player += 1
    return state_sys

@njit
def get_list_action_old(player_state_origin:np.int64):
    player_state = player_state_origin.copy()
    player_state = player_state.astype(np.int64)
    amount = player_state[2]
    index_between = int((12 - amount) + 3)
    card = player_state[3:index_between]
    list_action = card[np.where(card>= 0)[0]]
    list_card_player= np.where(player_state[index_between+2:index_between+int((12 - amount) + 2)] == 11)[0]
    if (12-amount)*3 < player_state[1]:
        list_action = np.array([13])

    if len(list_card_player) != 0 and player_state[-2] != 1 and len(list_action) > 1:
        list_action = np.append(list_action,np.array([12]))

    if player_state[-2] == 1:
        index = np.where(list_action == 11)[0][0]
        # print("INDEXXXXXXXXXXX:",index)
        list_action = np.delete(list_action, index)
    return np.unique(list_action)

@njit
def get_list_action(player_state_origin:np.int64):
    list_action_return = np.zeros(14)
    player_state = player_state_origin.copy()
    player_state = player_state.astype(np.int64)
    amount = player_state[2]
    index_between = int((12 - amount) + 3)
    card = player_state[3:index_between]
    list_action = card[np.where(card>= 0)[0]]
    list_card_player= np.where(player_state[index_between+2:index_between+int((12 - amount) + 2)] == 11)[0]
    if (12-amount)*3 < player_state[1]:
        list_action = np.array([13])

    if len(list_card_player) != 0 and player_state[-2] != 1 and len(list_action) > 1:
        list_action = np.append(list_action,np.array([12]))

    if player_state[-2] == 1:
        index = np.where(list_action == 11)[0][0]
        # print("INDEXXXXXXXXXXX:",index)
        list_action = np.delete(list_action, index)
    list_action_return[np.unique(list_action)] = 1
    return list_action_return

@njit
def reset_card_player(state_sys):
    amount_player = state_sys[2]
    for player in range(amount_player):
        index_player_s = (player % amount_player) * (12 - amount_player) + 3*amount_player*(12-amount_player) + (player % amount_player+1) *2 +3
        index_player_e = index_player_s+ (12 - amount_player)
        for i in range(index_player_s,index_player_e):
            state_sys[i] = -1
    return state_sys

@njit
def amount_action():
    return 14

@njit
def amount_player():
    return 5

@njit()
def amount_state():
    return 57

@njit
def test_action(player_state,action):
    amount = player_state[2]
    index_between = int((12 - amount) + 3)
    if action == 12:
        player_state = move_card(player_state,11,amount,index_between+2,index_between+int((12 - amount) + 2),3,index_between)
        player_state[-1] += 1
        player_state[-2] = 1
        return player_state
    # player_state[-2] = 0
    if player_state[-1] > 0:
        player_state = move_card(player_state,action,amount,3,index_between,index_between+2,index_between+int((12 - amount) + 2))
    player_state[-1] -= 1
    return player_state

@njit
def move_card(state,card,amount,start_1 = 0,end_1=0,start_2 = 0,end_2 = 0):
    index_relative_from = np.where(state[start_1:end_1] == card)[0]
    index_relative_to = np.where(state[start_2:end_2]==-1)[0]
    index_relative_from = index_relative_from[0] + start_1
    index_relative_to = index_relative_to[0] + start_2
    temp = state[index_relative_from]
    state[index_relative_from] = state[index_relative_to]
    state[index_relative_to] = temp
    return state
def one_game_print(list_player,per_file):
    amount_player = len(list_player)
    state_sys = reset(amount_player)
    temp_file = [[0] for i in range(amount_player)]
    amount_player = state_sys[2]
    turn = state_sys[1]

    while turn<(12-amount_player)*3:
        round = state_sys[0]-1
        turn = state_sys[1]
        print("Luot: ",turn,state_sys)
        list_action = [[-1,-1,-1] for i in range(amount_player)]
        for id_player in range(len(list_player)):
            player_state = get_player_state(state_sys,id_player)
            count = 0
            while player_state[-1] > 0:
                print(list_action[id_player])
                action, temp_file[id_player], per_file = list_player[id_player](player_state,temp_file[id_player],per_file)
                list_action[id_player][count] = action
                count += 1
                player_state = test_action(player_state,action)
            player_state = get_player_state(state_sys,id_player)
            print(id_player,player_state)
        list_action = np.array(list_action)
        print(list_action)
        state_sys = step(state_sys,list_action,amount_player,turn,round)
        if turn % (12-amount_player) == 0:
            state_sys = caculater_score(state_sys,amount_player)
            if state_sys[0] < 3:
                state_sys[0] += 1
                state_sys = reset_card_player(state_sys)
        if turn == (12-amount_player)*3:
            state_sys = caculator_pudding(state_sys,amount_player)
        if turn <= (12-amount_player)*3:
            state_sys[1] += 1
        print(state_sys)
    # print(state_sys)
    for id_player in range(len(list_player)):
        list_action[id_player], temp_file[id_player], per_file = list_player[id_player](get_player_state(state_sys,id_player),temp_file[id_player],per_file)    
    winner = winner_victory(state_sys)
    return winner,per_file

def one_game(list_player_,per_file):
    amount_player = len(list_player_)
    state_sys = reset(amount_player)
    temp_file = [[0] for i in range(amount_player)]
    amount_player = state_sys[2]
    turn = state_sys[1]

    while turn<(12-amount_player)*3:
        round = state_sys[0]-1
        turn = state_sys[1]
        # print("Luot: ",turn,state_sys)
        list_action = [[-1,-1,-1] for i in range(amount_player)]
        for id_player in range(len(list_player_)):
            player_state = get_player_state(state_sys,id_player)
            count = 0
            while player_state[-1] > 0:
                # print(list_action[id_player])
                action, temp_file[id_player], per_file = list_player_[id_player](player_state,temp_file[id_player],per_file)
                list_action[id_player][count] = action
                count += 1
                player_state = test_action(player_state,action)
            player_state = get_player_state(state_sys,id_player)
            # print(id_player,player_state)
        list_action = np.array(list_action)
        # print(list_action)
        state_sys = step(state_sys,list_action,amount_player,turn,round)
        if turn % (12-amount_player) == 0:
            state_sys = caculater_score(state_sys,amount_player)
            if state_sys[0] < 3:
                state_sys[0] += 1
                state_sys = reset_card_player(state_sys)
        if turn == (12-amount_player)*3:
            state_sys = caculator_pudding(state_sys,amount_player)
        if turn <= (12-amount_player)*3:
            state_sys[1] += 1
        # print(state_sys)
    # print(state_sys)
    for id_player in range(len(list_player_)):
        list_action[id_player], temp_file[id_player], per_file = list_player_[id_player](get_player_state(state_sys,id_player),temp_file[id_player],per_file)    
    winner = winner_victory(state_sys)
    return winner,per_file

def player_random(player_state,file_temp,file_per):
    a = get_list_action(player_state)
    b = random.randint(0,len(a)-1)
    return a[b],file_temp,file_per

def normal_main(list_player,amount_game,file_per):
    amount_player = len(list_player)
    player_list_index = [ i for i in range(amount_player)]
    num_won = [0 for i in range(amount_player)]
    for game in range(amount_game):
        random.shuffle(player_list_index)
        list_player_shuffle = [list_player[i] for i in player_list_index]
        winner, file_per = one_game(list_player_shuffle,file_per)
        # print(list_player_shuffle, winner, player_list_index)
        for win in winner:
            num_won[player_list_index[win]] += 1
    return num_won,file_per

def normal_main_print(list_player,amount_game,file_per):
    amount_player = len(list_player)
    player_list_index = [ i for i in range(amount_player)]
    num_won = [0 for i in range(amount_player)]
    for game in range(amount_game):
        random.shuffle(player_list_index)
        list_player_shuffle = [list_player[i] for i in player_list_index]
        winner, file_per = one_game_print(list_player_shuffle,file_per)
        for win in winner:
            num_won[player_list_index[win]] += 1
    return num_won,file_per


@njit()
def numba_one_game(p_lst_idx_shuffle, p0, p1, p2, p3, p4,per_file ):
    amount_player = 5
    state_sys = reset(amount_player)

    temp_1_player = List()
    temp_1_player.append(np.array([[0.]]))
    temp_file = [temp_1_player]*(5)

    amount_player = state_sys[2]
    turn = state_sys[1]

    while turn<(12-amount_player)*3:
        round = state_sys[0]-1
        turn = state_sys[1]
        # print("Luot: ",turn,state_sys)
        list_action = [[-1,-1,-1] for i in range(amount_player)]
        for id_player in range(amount_player):
            player_state = get_player_state(state_sys,id_player)
            count = 0
            while player_state[-1] > 0:
                # print(list_action[id_player])
                if p_lst_idx_shuffle[id_player] == 0:
                    act, temp_file[id_player], per_file = p0(player_state, temp_file[id_player], per_file)
                elif p_lst_idx_shuffle[id_player] == 1:
                    act, temp_file[id_player], per_file = p1(player_state, temp_file[id_player], per_file)
                elif p_lst_idx_shuffle[id_player] == 2:
                    act, temp_file[id_player], per_file = p2(player_state, temp_file[id_player], per_file)
                elif p_lst_idx_shuffle[id_player] == 3:
                    act, temp_file[id_player], per_file = p3(player_state, temp_file[id_player], per_file)
                else:
                    act, temp_file[id_player], per_file = p4(player_state, temp_file[id_player], per_file)
                list_action[id_player][count] = act
                count += 1
                player_state = test_action(player_state,act)
            player_state = get_player_state(state_sys,id_player)
            # print(id_player,player_state)
        list_action = np.array(list_action)
        # print(list_action)
        state_sys = step(state_sys,list_action,amount_player,turn,round)
        if turn % (12-amount_player) == 0:
            state_sys = caculater_score(state_sys,amount_player)
            if state_sys[0] < 3:
                state_sys[0] += 1
                state_sys = reset_card_player(state_sys)
        if turn == (12-amount_player)*3:
            state_sys = caculator_pudding(state_sys,amount_player)
        if turn <= (12-amount_player)*3:
            state_sys[1] += 1
        # print(state_sys)
    # print(state_sys)
    for id_player in range(5):
        p_state = get_player_state(state_sys,id_player)
        if p_lst_idx_shuffle[id_player] == 0:
            act, temp_file[id_player], per_file = p0(p_state, temp_file[id_player], per_file)
        elif p_lst_idx_shuffle[id_player] == 1:
            act, temp_file[id_player], per_file = p1(p_state, temp_file[id_player], per_file)
        elif p_lst_idx_shuffle[id_player] == 2:
            act, temp_file[id_player], per_file = p2(p_state, temp_file[id_player], per_file)
        elif p_lst_idx_shuffle[id_player] == 3:
            act, temp_file[id_player], per_file = p3(p_state, temp_file[id_player], per_file)
        else:
            act, temp_file[id_player], per_file = p4(p_state, temp_file[id_player], per_file)    
    winner = winner_victory(state_sys)
    return winner,per_file

@njit()
def numba_main(p0, p1, p2, p3, p4, num_game,per_file):
    num_won = np.zeros(amount_player())
    p_lst_idx = np.array([0,1,2,3,4])
    for _n in range(num_game):
        np.random.shuffle(p_lst_idx)
        winner, per_file = numba_one_game(p_lst_idx, p0, p1, p2, p3, p4,per_file )
        for win in winner:
            num_won[p_lst_idx[win]] += 1
    return list(num_won.astype(np.int64)), per_file

























    
from system.mainFunc import dict_game_for_player, load_data_per2
game_name_ = 'SushiGo'
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
def test2_An_130922(p_state, file_per_2):
    list_action = get_list_action(p_state)
    list_action = np.where(list_action == 1)[0]
    action = neural_network_an_130922(p_state, file_per_2, list_action)
    return action

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
def get_func(player_state, id, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11, per12, per13, per14, per15):
    if id == 0: return test2_An_270922(player_state, per0)
    elif id == 1: return test2_Dat_130922(player_state, per1)
    elif id == 2: return test2_Hieu_270922(player_state, per2)
    elif id == 3: return test2_Khanh_270922(player_state, per3)
    elif id == 4: return test2_Phong_130922(player_state, per4)
    elif id == 5: return test2_An_200922(player_state, per5)
    elif id == 6: return test2_Phong_130922(player_state, per6)
    elif id == 7: return test2_Dat_130922(player_state, per7)
    elif id == 8: return test2_Khanh_200922(player_state, per8)
    elif id == 9: return test2_NhatAnh_200922(player_state, per9)
    elif id == 10: return test2_Hieu_130922(player_state, per10)
    elif id == 11: return test2_Phong_130922(player_state, per11)
    elif id == 12: return test2_Khanh_130922(player_state, per12)
    elif id == 13: return test2_Dat_130922(player_state, per13)
    elif id == 14: return test2_NhatAnh_130922(player_state, per14)
    else: return test2_An_130922(player_state, per15)

@njit()
def one_game_numba(p0, list_other, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11, per12, per13, per14, per15):
    amount_player = 5
    state_sys = reset(amount_player)
    amount_player = state_sys[2]
    turn = state_sys[1]

    _temp_ = List()
    _temp_.append(np.array([[0]]))


    while turn<(12-amount_player)*3:
        round = state_sys[0]-1
        turn = state_sys[1]
        list_action = [[-1,-1,-1] for i in range(amount_player)]
        for id_player in range(5):
            player_state = get_player_state(state_sys,id_player)
            count = 0
            while player_state[-1] > 0:
                if list_other[id_player] == -1:
                    action, _temp_, per_player = p0(player_state,_temp_,per_player)
                else:
                    action = get_func(player_state, list_other[id_player], per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11, per12, per13, per14, per15)
                list_action[id_player][count] = action
                count += 1
                player_state = test_action(player_state,action)
            player_state = get_player_state(state_sys,id_player)
        list_action = np.array(list_action)
        state_sys = step(state_sys,list_action,amount_player,turn,round)
        if turn % (12-amount_player) == 0:
            state_sys = caculater_score(state_sys,amount_player)
            if state_sys[0] < 3:
                state_sys[0] += 1
                state_sys = reset_card_player(state_sys)
        if turn == (12-amount_player)*3:
            state_sys = caculator_pudding(state_sys,amount_player)
        if turn <= (12-amount_player)*3:
            state_sys[1] += 1

    for idx in range(5):
        if list_other[idx] == -1:
            act, _temp_, per_player = p0(get_player_state(state_sys,idx), _temp_, per_player)
    winner = False
    winner_player = winner_victory(state_sys)
    if np.where(list_other == -1)[0] in  winner_player: winner = True
    else: winner = False
    return winner,  per_player


@njit()
def n_game_numba(p0, num_game, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11, per12, per13, per14, per15):
    win = 0
    for _n in range(num_game):
        list_other = np.append(np.random.choice(np.arange(16), (amount_player() - 1)), -1)
        np.random.shuffle(list_other)
        winner,per_player  = one_game_numba(p0, list_other, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11, per12, per13, per14, per15)
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
    per9 = list_data[9]
    per10 = list_data[10]
    per11 = list_data[11]
    per12 = list_data[12]
    per13 = list_data[13]
    per14 = list_data[14]
    per15 = list_data[15]
    return n_game_numba(p0, n_game, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11, per12, per13, per14, per15)





# @njit()
def one_game_numba_2(p0, list_other, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11, per12, per13, per14, per15):
    amount_player = 5
    state_sys = reset(amount_player)
    amount_player = state_sys[2]
    turn = state_sys[1]

    _temp_ = List()
    _temp_.append(np.array([[0]]))


    while turn<(12-amount_player)*3:
        round = state_sys[0]-1
        turn = state_sys[1]
        list_action = [[-1,-1,-1] for i in range(amount_player)]
        for id_player in range(5):
            player_state = get_player_state(state_sys,id_player)
            count = 0
            while player_state[-1] > 0:
                if list_other[id_player] == -1:
                    action, _temp_, per_player = p0(player_state,_temp_,per_player)
                else:
                    action = get_func(player_state, list_other[id_player], per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11, per12, per13, per14, per15)
                list_action[id_player][count] = action
                count += 1
                player_state = test_action(player_state,action)
            player_state = get_player_state(state_sys,id_player)
        list_action = np.array(list_action)
        state_sys = step(state_sys,list_action,amount_player,turn,round)
        if turn % (12-amount_player) == 0:
            state_sys = caculater_score(state_sys,amount_player)
            if state_sys[0] < 3:
                state_sys[0] += 1
                state_sys = reset_card_player(state_sys)
        if turn == (12-amount_player)*3:
            state_sys = caculator_pudding(state_sys,amount_player)
        if turn <= (12-amount_player)*3:
            state_sys[1] += 1

    for idx in range(5):
        if list_other[idx] == -1:
            act, _temp_, per_player = p0(get_player_state(state_sys,idx), _temp_, per_player)
    winner = False
    winner_player = winner_victory(state_sys)
    if np.where(list_other == -1)[0] in  winner_player: winner = True
    else: winner = False
    return winner,  per_player


# @njit()
def n_game_numba_2(p0, num_game, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11, per12, per13, per14, per15):
    win = 0
    for _n in range(num_game):
        list_other = np.append(np.random.choice(np.arange(16), (amount_player() - 1)), -1)
        np.random.shuffle(list_other)
        winner,per_player  = one_game_numba_2(p0, list_other, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11, per12, per13, per14, per15)
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
    per9 = list_data[9]
    per10 = list_data[10]
    per11 = list_data[11]
    per12 = list_data[12]
    per13 = list_data[13]
    per14 = list_data[14]
    per15 = list_data[15]
    return n_game_numba_2(p0, n_game, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11, per12, per13, per14, per15)
