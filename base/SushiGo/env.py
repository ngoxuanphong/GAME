from operator import index
import numpy as np
import random
import multiprocessing as mp
import time
from numba import jit, njit

@njit
def initEnv(n):
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
def getAgentState(state_sys,index_player):
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
def getReward(state_player):
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
def stepEnv(state_sys,list_action,amount_player,turn,round):
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
def reset_card_player(state_sys):
    amount_player = state_sys[2]
    for player in range(amount_player):
        index_player_s = (player % amount_player) * (12 - amount_player) + 3*amount_player*(12-amount_player) + (player % amount_player+1) *2 +3
        index_player_e = index_player_s+ (12 - amount_player)
        for i in range(index_player_s,index_player_e):
            state_sys[i] = -1
    return state_sys



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
    state_sys = initEnv(amount_player)
    temp_file = [[0] for i in range(amount_player)]
    amount_player = state_sys[2]
    turn = state_sys[1]

    while turn<(12-amount_player)*3:
        round = state_sys[0]-1
        turn = state_sys[1]
        print("Luot: ",turn,state_sys)
        list_action = [[-1,-1,-1] for i in range(amount_player)]
        for id_player in range(len(list_player)):
            player_state = getAgentState(state_sys,id_player)
            count = 0
            while player_state[-1] > 0:
                print(list_action[id_player])
                action, per_file = list_player[id_player](player_state,per_file)
                list_action[id_player][count] = action
                count += 1
                player_state = test_action(player_state,action)
            player_state = getAgentState(state_sys,id_player)
            print(id_player,player_state)
        list_action = np.array(list_action)
        print(list_action)
        state_sys = stepEnv(state_sys,list_action,amount_player,turn,round)
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
        list_action[id_player], per_file = list_player[id_player](getAgentState(state_sys,id_player),per_file)    
    winner = winner_victory(state_sys)
    return winner,per_file

def one_game(list_player_,per_file):
    amount_player = len(list_player_)
    state_sys = initEnv(amount_player)
    temp_file = [[0] for i in range(amount_player)]
    amount_player = state_sys[2]
    turn = state_sys[1]

    while turn<(12-amount_player)*3:
        round = state_sys[0]-1
        turn = state_sys[1]
        # print("Luot: ",turn,state_sys)
        list_action = [[-1,-1,-1] for i in range(amount_player)]
        for id_player in range(len(list_player_)):
            player_state = getAgentState(state_sys,id_player)
            count = 0
            while player_state[-1] > 0:
                # print(list_action[id_player])
                action, per_file = list_player_[id_player](player_state,per_file)
                list_action[id_player][count] = action
                count += 1
                player_state = test_action(player_state,action)
            player_state = getAgentState(state_sys,id_player)
            # print(id_player,player_state)
        list_action = np.array(list_action)
        # print(list_action)
        state_sys = stepEnv(state_sys,list_action,amount_player,turn,round)
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
        list_action[id_player], per_file = list_player_[id_player](getAgentState(state_sys,id_player),per_file)    
    winner = winner_victory(state_sys)
    return winner,per_file

def player_random(player_state,file_temp,file_per):
    a = getValidActions(player_state)
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
    state_sys = initEnv(amount_player)

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
            player_state = getAgentState(state_sys,id_player)
            count = 0
            while player_state[-1] > 0:
                # print(list_action[id_player])
                if p_lst_idx_shuffle[id_player] == 0:
                    act, per_file = p0(player_state, per_file)
                elif p_lst_idx_shuffle[id_player] == 1:
                    act, per_file = p1(player_state, per_file)
                elif p_lst_idx_shuffle[id_player] == 2:
                    act, per_file = p2(player_state, per_file)
                elif p_lst_idx_shuffle[id_player] == 3:
                    act, per_file = p3(player_state, per_file)
                else:
                    act, per_file = p4(player_state, per_file)
                list_action[id_player][count] = act
                count += 1
                player_state = test_action(player_state,act)
            player_state = getAgentState(state_sys,id_player)
            # print(id_player,player_state)
        list_action = np.array(list_action)
        # print(list_action)
        state_sys = stepEnv(state_sys,list_action,amount_player,turn,round)
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
        p_state = getAgentState(state_sys,id_player)
        if p_lst_idx_shuffle[id_player] == 0:
            act, per_file = p0(p_state, per_file)
        elif p_lst_idx_shuffle[id_player] == 1:
            act, per_file = p1(p_state, per_file)
        elif p_lst_idx_shuffle[id_player] == 2:
            act, per_file = p2(p_state, per_file)
        elif p_lst_idx_shuffle[id_player] == 3:
            act, per_file = p3(p_state, per_file)
        else:
            act, per_file = p4(p_state, per_file)    
    winner = winner_victory(state_sys)
    return winner,per_file

@njit()
def numba_main(p0, p1, p2, p3, p4, num_game,per_file):
    num_won = np.zeros(getAgentSize())
    p_lst_idx = np.array([0,1,2,3,4])
    for _n in range(num_game):
        np.random.shuffle(p_lst_idx)
        winner, per_file = numba_one_game(p_lst_idx, p0, p1, p2, p3, p4,per_file )
        for win in winner:
            num_won[p_lst_idx[win]] += 1
    return list(num_won.astype(np.int64)), per_file


from numba.typed import List
from base.SushiGo.AgentEnv import *

@njit() 
def test2_Phong_130922_New(state,file_per_2):
    id_model = int(file_per_2[-1][0][0])
    action = file_temp_to_action_Phong_130922(state, file_per_2[id_model])
    if getReward(state) == 0:
        file_per_2[-1][0][0] = np.random.choice(np.array([0., 1., 2.])) 
    return action

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
    elif id == 11: return test2_Phong_130922_New(player_state, per11)
    elif id == 12: return test2_Khanh_130922(player_state, per12)
    elif id == 13: return test2_Dat_130922(player_state, per13)
    elif id == 14: return test2_NhatAnh_130922(player_state, per14)
    else: return test2_An_130922(player_state, per15)

@njit()
def one_game_numba(p0, list_other, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11, per12, per13, per14, per15):
    amount_player = 5
    state_sys = initEnv(amount_player)
    amount_player = state_sys[2]
    turn = state_sys[1]



    while turn<(12-amount_player)*3:
        round = state_sys[0]-1
        turn = state_sys[1]
        list_action = [[-1,-1,-1] for i in range(amount_player)]
        for id_player in range(5):
            player_state = getAgentState(state_sys,id_player)
            count = 0
            while player_state[-1] > 0:
                if list_other[id_player] == -1:
                    action, per_player = p0(player_state,per_player)
                elif list_other[id_player] == -2:
                    action = random_Env(player_state)
                else:
                    action = get_func(player_state, list_other[id_player], per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11, per12, per13, per14, per15)
                list_action[id_player][count] = action
                count += 1
                player_state = test_action(player_state,action)
            player_state = getAgentState(state_sys,id_player)
        list_action = np.array(list_action)
        state_sys = stepEnv(state_sys,list_action,amount_player,turn,round)
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
            act, per_player = p0(getAgentState(state_sys,idx), per_player)
        else:
            action = get_func(getAgentState(state_sys,idx), list_other[idx], per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11, per12, per13, per14, per15)
    winner = False
    winner_player = winner_victory(state_sys)
    if np.where(list_other == -1)[0] in  winner_player: winner = True
    else: winner = False
    return winner,  per_player


@njit()
def n_game_numba(p0, num_game, per_player, level, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11, per12, per13, per14, per15):
    win = 0
    if level == 0:
        list_other = np.array([-2, -2, -2, -2, -1])
    elif level == 1:
        list_other = np.array([10, 8, 7, 0, -1])
    elif level == 2:
        list_other = np.array([2, 1, 12, 13, -1])
    elif level == 3:
        list_other = np.array([3, 11, 15, 5, -1])
    else:
        raise Exception('Hiện tại không có level này')
    for _n in range(num_game):
        np.random.shuffle(list_other)
        winner,per_player  = one_game_numba(p0, list_other, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11, per12, per13, per14, per15)
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
    per9 = list_data[9]
    per10 = list_data[10]
    per11 = list_data[11]
    per12 = list_data[12]
    per13 = list_data[13]
    per14 = list_data[14]
    per15 = list_data[15]
    return n_game_numba(p0, n_game, per_player, level, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11, per12, per13, per14, per15)


# @njit()
def one_game_numba_2(p0, list_other, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11, per12, per13, per14, per15):
    amount_player = 5
    state_sys = initEnv(amount_player)
    amount_player = state_sys[2]
    turn = state_sys[1]



    while turn<(12-amount_player)*3:
        round = state_sys[0]-1
        turn = state_sys[1]
        list_action = [[-1,-1,-1] for i in range(amount_player)]
        for id_player in range(5):
            player_state = getAgentState(state_sys,id_player)
            count = 0
            while player_state[-1] > 0:
                if list_other[id_player] == -1:
                    action, per_player = p0(player_state,per_player)
                elif list_other[id_player] == -2:
                    action = random_Env(player_state)
                else:
                    action = get_func(player_state, list_other[id_player], per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11, per12, per13, per14, per15)
                list_action[id_player][count] = action
                count += 1
                player_state = test_action(player_state,action)
            player_state = getAgentState(state_sys,id_player)
        list_action = np.array(list_action)
        state_sys = stepEnv(state_sys,list_action,amount_player,turn,round)
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
            act, per_player = p0(getAgentState(state_sys,idx), per_player)
        else:
            action = get_func(getAgentState(state_sys,idx), list_other[idx], per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11, per12, per13, per14, per15)
    winner = False
    winner_player = winner_victory(state_sys)
    if np.where(list_other == -1)[0] in  winner_player: winner = True
    else: winner = False
    return winner,  per_player


# @njit()
def n_game_numba_2(p0, num_game, per_player, level, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11, per12, per13, per14, per15):
    win = 0
    if level == 0:
        list_other = np.array([-2, -2, -2, -2, -1])
    elif level == 1:
        list_other = np.array([10, 8, 7, 0, -1])
    elif level == 2:
        list_other = np.array([2, 1, 12, 13, -1])
    elif level == 3:
        list_other = np.array([3, 11, 15, 5, -1])
    else:
        raise Exception('Hiện tại không có level này')
    for _n in range(num_game):
        np.random.shuffle(list_other)
        winner,per_player  = one_game_numba_2(p0, list_other, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11, per12, per13, per14, per15)
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
    per9 = list_data[9]
    per10 = list_data[10]
    per11 = list_data[11]
    per12 = list_data[12]
    per13 = list_data[13]
    per14 = list_data[14]
    per15 = list_data[15]
    return n_game_numba_2(p0, n_game, per_player, level, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11, per12, per13, per14, per15)




