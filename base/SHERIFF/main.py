import pandas as pd
import numpy as np
import re
import numba
from numba import vectorize, jit, cuda, float64, njit, prange
import os


@njit(fastmath=True, cache=True)
def reset():
    normal_card = np.concatenate((np.full(48, 1), np.full(36, 2), np.full(36, 3),np.full(24, 4), np.full(22, 5), 
                              np.full(21, 6), np.full(12, 7), np.full(5, 8)))
    royal_card = np.concatenate((np.full(2, 9), np.full(2, 10), np.full(2, 11), np.full(2, 12), np.full(2, 13), 
                                 np.full(1, 14), np.full(1, 15)))
    all_penalty = np.array([2, 2, 2, 2, 4, 4, 4, 4, 3, 4, 4, 4, 4, 5, 5])
    all_reward = np.array([2, 3 ,3 ,4 ,6 ,7 ,8 ,9 ,4 ,6 ,6 ,8 ,6 ,9 ,9])
    start_player = np.concatenate((np.array([50]), np.full(95, 0)))
    np.random.shuffle(normal_card)
    environment = np.array([0])
    for i in prange(4):
        player_i = np.copy(start_player)
        if i == 0:
            player_i[1] = 1 
            card_player = player_i[-15:]
            for card in normal_card[i*5:(i+1)*5]:
                card_player[card-1] += 1
            player_i[-15:] = card_player
            environment = player_i
        else:
            card_player = player_i[-15:]
            for card in normal_card[i*5:(i+1)*5]:
                card_player[card-1] += 1
            player_i[-15:] = card_player
            environment = np.append(environment, player_i)
    all_card = np.concatenate((normal_card[20:], royal_card))
    np.random.shuffle(all_card)
    left_up = np.array([0]*125)
    left_up[:5] = all_card[20:25]
    right_up = np.array([0]*125)
    right_up[:5] = all_card[25:30]
    down_card = all_card[30:150]
    temp_drop = np.zeros(15)
    environment = np.concatenate((environment, down_card, left_up, right_up, temp_drop))
    environment = np.append(environment, np.array([-0.5, 0, 1, 1, 1]))
    #last_checked, number_checked,  id_action, round , phase

    return environment

@njit(fastmath=True, cache=True)
def state_to_player(env_state):
    player_action = env_state[-3]
    player_state = env_state[96*player_action:96*(player_action+1)]
    for idx in range(1, 4):
        id = (player_action + idx)%4
        all_other_player_in4 = env_state[96*id:96*(id+1)]
        other_player_in4 = all_other_player_in4[:25]
        player_state = np.append(player_state, other_player_in4)
    player_state = np.concatenate((player_state, env_state[504:519], env_state[629:644], env_state[-2:]))
    return player_state

def player_random(player_state, file_temp, file_per):
    list_action = get_list_action(player_state)
    action = int(np.random.choice(list_action))
    if check_victory(player_state) == -1:
        # print('chưa hết game')
        pass
    else:
        if check_victory(player_state) == 1:
            # print('win')
            pass
        else:
            # print('lose')
            pass

    return action, file_temp, file_per

@njit(fastmath=True, cache=True)
def get_list_action(player_state):
    # player_action = int(player_state[-3])
    phase_env = player_state[-1]
    player_state_own = player_state[:96]
    if phase_env == 1:
        list_action = np.array([15])
        player_card = player_state_own[-15:]
        for id in range(15):
            if player_card[id] == 0:
                continue
            else:
                list_action = np.append(list_action, id)
        return list_action 

    elif phase_env == 2:
        #lấy thẻ từ các chồng bài: 16-bài rút, 17-lật trái, 18-lật phải
        return np.array([16, 17, 18])    

    elif phase_env == 3:
        #trả thẻ bỏ vào  chồng bài lật: 19: lật trái, 20: lật phải
        return np.array([19, 20])

    elif phase_env == 4:
        #lựa chọn bỏ qua thẻ
        list_action = np.array([36])
        player_card = player_state_own[-15:]
        for id in range(15):
            if player_card[id] == 0:
                continue
            else:
                list_action = np.append(list_action, id+21)
        if np.sum(player_state_own[-45:-30]) == 0:
            list_action = np.delete(list_action, 0)
        return list_action  

    elif phase_env == 5:
        #chọn 1 trong 4 loại hàng chính ngạch
        list_action = np.array([37, 38, 39, 40])
        return list_action

    elif phase_env == 6:
        #41, 42, 43 lần lượt là check người ở vị trí 1,2,3 sau mình, 44 là ko check nữa
        list_action = np.array([44])
        for id in range(3):
            #kiểm tra xem đã check hay chưa bằng cách xem mặt hàng khai báo
            type_bag_other_player = player_state[95+25*id+3]
            if type_bag_other_player != 0:
                list_action = np.append(list_action, 41+id)

        list_action = np.delete(list_action, 0)
        return list_action

    elif phase_env == 7:
        #45 là ko hối lộ nữa, 46 là thêm coin
        if player_state_own[0] > 0:
            list_action = np.array([45, 46])
        else:
            list_action = np.array([45])
        return list_action

    elif phase_env == 8:
        list_action = np.array([62])
        player_card_done = player_state_own[-75:-60]
        for id in range(15):
            if player_card_done[id] == 0:
                continue
            else:
                list_action = np.append(list_action, id+47)
        return list_action

    elif phase_env == 9:
        list_action = np.array([78])
        player_card_bag = player_state_own[-45:-30]
        for id in range(15):
            if player_card_bag[id] == 0:
                continue
            else:
                list_action = np.append(list_action, id+63)
        return list_action

    elif phase_env == 10:
        #79 là có check hàng, 80 là cho thoát
        list_action = np.array([79, 80])
        return list_action

    elif phase_env == 11:
        return np.array([81, 82])

@njit(fastmath=True, cache=True)
def step(env_state, action, all_penalty):
    phase_env = env_state[-1]
    id_action = int(env_state[-3])
    player_in4 = env_state[96*id_action:96*(id_action+1)]
    # print(player_in4)

    if phase_env == 1:
        if action != 15:
            player_card = player_in4[-15:]
            temp_drop_card = env_state[754:769]
            player_card[action] -= 1
            temp_drop_card[action] += 1
            player_in4[-15:] = player_card
            env_state[96*id_action:96*(id_action+1)] = player_in4
            env_state[754:769] = temp_drop_card
            if np.sum(player_card) == 0:
                env_state[-1] += 1
        else:
            env_state[-1] += 1

    elif phase_env == 2: 
        if action == 16:
            player_card = player_in4[-15:]
            # print('check', player_card)
            card_down = env_state[384:504]
            number_card_get = int(6 - np.sum(player_card))
            card_get = card_down[:number_card_get]
            if len(card_get) > 0:
                for card in card_get:
                    player_card[int(card)-1] += 1
            card_down = np.concatenate((card_down[number_card_get:], np.zeros(number_card_get)))
            env_state[384:504] = card_down
            player_in4[-15:] = player_card
            env_state[96*id_action:96*(id_action+1)] = player_in4
            env_state[-1] += 1
        else: 
            player_card = player_in4[-15:]
            card_up = env_state[504+125*(action-17):504+125*(action-16)]
            card_get = card_up[0]
            player_card[int(card_get)-1] += 1
            card_up = np.append(card_up[1:], 0)
            env_state[504+125*(action-17):504+125*(action-16)] = card_up
            player_in4[-15:] = player_card
            env_state[96*id_action:96*(id_action+1)] = player_in4
            if np.sum(player_card) == 6:
                env_state[-1] += 1

    elif phase_env == 3:
        temp_card_drop = env_state[754:769]
        card_up = env_state[504+125*(action-19):504+125*(action-18)]
        for i in range(len(temp_card_drop)):
            if temp_card_drop[i] > 0:
                card_up = np.append(np.array([i+1]*int(temp_card_drop[i])), card_up)
        if np.sum(temp_card_drop) > 0:
            card_up = card_up[:int(-np.sum(temp_card_drop))]
        temp_card_drop = np.zeros(15)
        env_state[754:769] = temp_card_drop
        env_state[504+125*(action-19):504+125*(action-18)] = card_up
        env_state[-1] = 1
        env_state[-3] = (env_state[-3] + 1)%4
        if env_state[-3] == (env_state[-2] - 1)%4:
            env_state[-3] = (env_state[-3] + 1)%4
            env_state[-1] = 4

    elif phase_env == 4:
        card_bag = player_in4[-45:-30]
        player_card = player_in4[-15:]
        if action == 36:
            if np.sum(card_bag) == 0:
                print('T TOANG NÀY')
                return env_state
            else:
                env_state[-3] = (env_state[-3] + 1)%4
                if env_state[-3] == (env_state[-2] - 1)%4:
                    env_state[-3] = (env_state[-3] + 1)%4
                    env_state[-1] +=1
        else:
            card_bag[action-21] += 1
            player_card[action-21] -= 1
            player_in4[-15:] = player_card
            player_in4[-45:-30] = card_bag
            env_state[96*id_action:96*(id_action+1)] = player_in4
            if np.sum(card_bag) == 5:
                env_state[-3] = (env_state[-3] + 1)%4
                if env_state[-3] == (env_state[-2] - 1)%4:
                    env_state[-3] = (env_state[-3] + 1)%4
                    env_state[-1] +=1

    elif phase_env == 5:
        type_bag = action - 36
        player_in4[2] = type_bag
        env_state[96*id_action:96*(id_action+1)] = player_in4
        env_state[-3] = (env_state[-3] + 1)%4
        if env_state[-3] == (env_state[-2] - 1)%4:
            env_state[-1] +=1

    elif phase_env == 6:
        player_checked = action-40 
        env_state[-3] = (env_state[-3] + player_checked)%4
        env_state[-4] += 1
        env_state[-1] +=1
        env_state[-5] = env_state[-3]

    elif phase_env == 7:
        if action == 46:
            player_in4[0] -= 1
            #new: chỉnh vị trí coin hối lộ
            player_in4[3] += 1
            env_state[96*id_action:96*(id_action+1)] = player_in4
        else:
            env_state[-1] += 1

    elif phase_env == 8:
        if action == 62:
            env_state[-1] += 1
        else:
            all_card_done = player_in4[-75:-60]
            all_card_bride = player_in4[-90:-75]
            card_bride = action-47
            all_card_done[card_bride] -= 1
            all_card_bride[card_bride] += 1
            player_in4[-75:-60] = all_card_done
            player_in4[-90:-75] = all_card_bride
            env_state[96*id_action:96*(id_action+1)] = player_in4
            # env_state[-1] += 1

    elif phase_env == 9:
        if action == 78:
            env_state[-1] += 1
            env_state[-3] = (env_state[-2] - 1)%4
        else:
            # print(player_in4[-90:-60])
            all_card_bag = player_in4[-45:-30]
            all_card_bride_bag = player_in4[-60:-45]
            card_bride = action-63
            all_card_bag[card_bride] -= 1
            all_card_bride_bag[card_bride] += 1
            #cập nhập số thẻ trong túi dùng hối lộ
            player_in4[5] += 1
            player_in4[-45:-30] = all_card_bag
            player_in4[-60:-45] = all_card_bride_bag
            env_state[96*id_action:96*(id_action+1)] = player_in4
            # env_state[-1] += 1
            # env_state[-3] = (env_state[-2] - 1)%4

    elif phase_env == 10:
        id_checked = int(env_state[-5])
        player_checked = env_state[96*id_checked:96*(id_checked+1)]
        type_bag = player_checked[2]
        if action == 79:
            #có check
            #B1: khôi phục coin
            player_checked[0] += player_checked[3]
            player_checked[3] = 0
            #B3 khôi phục thẻ done
            player_checked[-75:-60] = player_checked[-75:-60] + player_checked[-90:-75]
            player_checked[-90:-75] = np.zeros(15)
            #khôi phục thẻ trong túi
            player_checked[-45:-30] = player_checked[-45:-30] + player_checked[-60:-45]
            player_checked[-60:-45] = np.zeros(15)
            #tính toán cho sheriff
            player_checked_card_bag = player_checked[-45:-30]
            player_checked_card_done = player_checked[-75:-60]
            card_bag_drop = player_in4[-30:-15]
            #cập nhật coin
            if player_checked_card_bag[int(type_bag)-1] == np.sum(player_checked_card_bag):
                penalty = -player_checked_card_bag[int(type_bag)-1]*all_penalty[int(type_bag)-1]
            else:
                penalty = np.sum(np.multiply(player_checked_card_bag, all_penalty)) - player_checked_card_bag[int(type_bag)-1]*all_penalty[int(type_bag)-1]
            player_in4[0] += penalty
            player_checked[0] -= penalty
            #cập nhật thẻ
            for id in range(15):
                if id == type_bag - 1:
                    player_checked_card_done[id] += player_checked_card_bag[id]
                else:
                    card_bag_drop[id] += player_checked_card_bag[id]
            player_checked[-45:-30] = np.zeros(15)
            player_checked[-75:-60] = player_checked_card_done
            player_checked[5] = 0
            player_checked[2] = 0
            player_checked[4] = np.sum(player_checked_card_done[4:])
            env_state[96*id_checked:96*(id_checked+1)] = player_checked
            player_in4[-30:-15] = card_bag_drop
            env_state[96*id_action:96*(id_action+1)] = player_in4
            #cập nhật trong hệ thống
            if np.sum(card_bag_drop) > 0:
                env_state[-1] += 1
                env_state[-3] = (env_state[-2] - 1)%4
            else:
                if env_state[-4] == 3:
                    env_state[-1] = 1
                    env_state[-3] = (env_state[-2])%4
                    env_state[-2] += 1
                    if env_state[-3] == (env_state[-2]-1)%4:
                        env_state[-3] = (env_state[-3] + 1)%4
                    env_state[-4] = 0
                else:
                    env_state[-1] = 6
                    env_state[-3] = (env_state[-2] - 1)%4
        else:
            #không check
            #B1: Cập nhật coin
            player_in4[0] += player_checked[3]
            # player_checked[0] -= player_checked[3]
            player_checked[3] = 0
            #B3: cập nhật thẻ done sheriff and player_checked
            # player_checked[-75:-60] = player_checked[-75:-60] - player_checked[-90:-75]
            player_in4[-75:-60] = player_in4[-75:-60] + player_checked[-90:-75]
            player_checked[-90:-75] = np.zeros(15)
            #cập nhật thẻ done từ hối lộ thẻ trong túi và cập nhật thẻ trong túi
            # player_checked[-45:-30] = player_checked[-45:-30] - player_checked[-60:-45]     #trừ thẻ trong túi đi
            player_in4[-75:-60] = player_in4[-75:-60] + player_checked[-60:-45]
            player_checked[-60:-45] = np.zeros(15)
            #cập nhật thẻ done player_checked
            player_checked[-75:-60] = player_checked[-75:-60] + player_checked[-45:-30]
            player_checked[-45:-30] = np.zeros(15)
            #cập nhật full thông tin
            player_checked[5] = 0
            player_checked[2] = 0
            player_checked[4] = np.sum(player_checked[-75:-60][4:])
            player_in4[4] = np.sum(player_in4[-75:-60][4:])
            env_state[96*id_checked:96*(id_checked+1)] = player_checked
            env_state[96*id_action:96*(id_action+1)] = player_in4
            #cập nhật hệ thống lượt chơi
            if env_state[-4] == 3:
                env_state[-1] = 1
                env_state[-3] = (env_state[-2])%4
                env_state[-2] += 1
                if env_state[-3] == (env_state[-2]-1)%4:
                    env_state[-3] = (env_state[-3] + 1)%4
                env_state[-4] = 0
            else:
                env_state[-1] = 6
                env_state[-3] = (env_state[-2] - 1)%4

    elif phase_env == 11:
        temp_card_drop = player_in4[-30:-15]
        card_up = env_state[504+125*(action-81):504+125*(action-80)]
        for i in range(len(temp_card_drop)):
            if temp_card_drop[i] > 0:
                card_up = np.append(np.array([i+1]*int(temp_card_drop[i])), card_up)
        # if np.sum(temp_card_drop) > 0:
        card_up = card_up[:int(-np.sum(temp_card_drop))]
        temp_card_drop = np.zeros(15)
        player_in4[-30:-15] = temp_card_drop
        env_state[504+125*(action-81):504+125*(action-80)] = card_up
        env_state[96*id_action:96*(id_action+1)] = player_in4

        if env_state[-4] == 3:
            env_state[-1] = 1
            env_state[-3] = (env_state[-2])%4
            env_state[-2] += 1
            if env_state[-3] == (env_state[-2]-1)%4:
                env_state[-3] = (env_state[-3] + 1)%4
            env_state[-4] = 0
        else:
            env_state[-1] = 6
            env_state[-3] = (env_state[-2] - 1)%4
    # print(env_state[-4:])
    return env_state

@njit(fastmath=True, cache=True)
def amount_action_space():
    return 83

@njit(fastmath=True, cache=True)  
def system_check_end(env_state):
    if env_state[-2] <= 8:
        return False
    else:
        return True

def action_player(env_state,list_player,file_temp,file_per):
    current_player = int(env_state[-3])
    player_state = state_to_player(env_state)
    played_move,file_temp[current_player],file_per = list_player[current_player](player_state,file_temp[current_player],file_per)
    return played_move,file_temp,file_per

@njit(fastmath=True, cache=True)  
def check_winner(env_state):
    all_reward = np.array([2, 3 ,3 ,4 ,6 ,7 ,8 ,9 ,4 ,6 ,6 ,8 ,6 ,9 ,9])
    all_number_count = np.array([1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3])
    id_action_1 = 0
    all_number_type_card = np.zeros(16)
    all_done_card = np.zeros(60)
    all_player_coin = np.array([env_state[96*i] for i in range(4)])
    all_player_coin = all_player_coin + np.array([0, 0.1, 0.2, 0.3])
    for id_player in range(4):
        player_i = env_state[96*id_player:96*(id_player+1)]
        player_i_done = player_i[-75:-60]
        all_done_card[15*id_player:15*(id_player+1)] = player_i_done
        all_player_coin[id_player] = all_player_coin[id_player] + np.sum(np.multiply(player_i_done, all_reward))
        all_done_card_i = player_i_done*all_number_count
        all_number_type_card[id_player] = all_done_card_i[0] + all_done_card_i[8] + all_done_card_i[12]
        all_number_type_card[id_player+4] = all_done_card_i[1] + all_done_card_i[9] + all_done_card_i[13]
        all_number_type_card[id_player+8] = all_done_card_i[2] + all_done_card_i[10] + all_done_card_i[14]
        all_number_type_card[id_player+12] = all_done_card_i[3] + all_done_card_i[11] 
    reward_King = np.array([20, 15, 15, 10])
    reward_Queen = np.array([10, 10, 10, 5])
    for type_i in range(4):
        count_type = all_number_type_card[4*type_i:4*(type_i+1)]
        top1 = np.max(count_type)
        if top1 == 0:
            continue
        count_top1 = len(np.where(count_type == top1))
        if count_top1 > 1:
            reward_i = np.floor((reward_King[type_i] + reward_Queen[type_i])/count_top1)
            all_player_coin = np.where(count_type == top1, all_player_coin + reward_i, all_player_coin)
            continue
        else:
            reward_i = reward_King[type_i]
            all_player_coin = np.where(count_type == top1, all_player_coin + reward_i, all_player_coin)
            count_type = np.where(count_type == top1, 0, count_type)
            top2 = np.max(count_type)
            if top2 == 0:
                continue
            count_top2 = len(np.where(count_type == top2))
            if count_top2 > 1:
                reward_i = np.floor(reward_Queen[type_i]/count_top2)
                all_player_coin = np.where(count_type == top2, all_player_coin + reward_i, all_player_coin)
                continue
            else:
                reward_i = reward_Queen[type_i]
                all_player_coin = np.where(count_type == top2, all_player_coin + reward_i, all_player_coin)
    for id_player in range(4):
        env_state[96*id_player] = all_player_coin[id_player]

    all_player_coin_int = np.array([int(all_player_coin[i]) for i in range(len(all_player_coin))])
    temp_win = np.max(all_player_coin_int)
    number_win = np.where(all_player_coin_int == temp_win)
    if len(number_win[0]) == 1:
        return number_win[0][0]
    else:
        all_legal = np.zeros(4)
        for id in number_win[0]:
            all_legal[id] = np.sum(all_done_card[15*id:15*(id+1)][:4])
        temp_win = np.max(all_legal)
        number_win = np.where(all_legal == temp_win)
        if len(number_win[0]) == 1:
            return number_win[0][0]
        else:
            all_unlegal = np.zeros(4)
            for id in number_win[0]:
                all_unlegal[id] = np.sum(all_done_card[15*id:15*(id+1)][4:])
            temp_win = np.max(all_unlegal)
            number_win = np.where(all_unlegal == temp_win)
            if len(number_win[0]) == 1:
                return number_win[0][0]
            else:
                all_player_coin_last = np.zeros(4)
                all_player_coin_last[number_win] = all_player_coin[number_win]
                return np.argmax(all_player_coin_last)
    
@njit(fastmath=True, cache=True)
def check_victory(player_state):
    round = player_state[-2]
    if round < 9:
        return -1
    else:
        all_player_coin = np.array([int(player_state[0])] + [int(player_state[96+25*id]) for id in range(3)])
        temp_win = np.max(all_player_coin)
        number_win = np.where(all_player_coin == temp_win)
        if len(number_win[0]) == 1:
            if number_win[0][0] == 0:
                return 1
            else:
                return 0
        else:
            all_legal = np.array([np.sum(player_state[21:25])] + [np.sum(player_state[96+25*id:96+25*(id+1)][-4:]) for id in range(3)])
            temp_win = np.max(all_legal)
            number_win = np.where(all_legal == temp_win)
            if len(number_win[0]) == 1:
                if number_win[0][0] == 0:
                    return 1
                else:
                    return 0
            else:
                all_unlegal = np.array([player_state[4]] + [player_state[96 + id*25 + 4] for id in range(3)])
                temp_win = np.max(all_unlegal)
                number_win = np.where(all_unlegal == temp_win)
                if len(number_win[0]) == 1:
                    if number_win[0][0] == 0:
                        return 1
                    else:
                        return 0
                else:
                    all_player_coin = np.array([player_state[0]] + [player_state[96+25*id] for id in range(3)])
                    all_player_coin_last = np.zeros(4) 
                    all_player_coin_last[number_win] = all_player_coin[number_win]
                    winner = np.argmax(all_player_coin_last)
                    if winner == 0:
                        return 1
                    else:
                        return 0
               
def one_game(list_player, file_temp, file_per, all_penalty):
    env_state = reset()
    while not system_check_end(env_state):
        # player_state = state_to_player(env_state)
        action, file_temp, file_per = action_player(env_state,list_player,file_temp,file_per)
        env_state = step(env_state, action, all_penalty)
    winner = check_winner(env_state)
    for id_player in range(4):
        env_state[-1] = 1
        id_action = env_state[-3]
        player_state = state_to_player(env_state)
        action, file_temp, file_per = action_player(env_state,list_player,file_temp,file_per)
        env_state[-3] = (env_state[-3] + 1)%4
    return winner, file_per

def normal_main(list_player, times, print_mode):
    count = np.zeros(len(list_player))
    file_per = [0]
    all_penalty = np.array([2, 2, 2, 2, 4, 4, 4, 4, 3, 4, 4, 4, 4, 5, 5])
    # count_game = np.zeros(10)
    all_id_player = np.arange(len(list_player))
    for van in range(times):
        # print(van)
        shuffle = np.random.choice(all_id_player, 4, replace=False)
        # for id in shuffle:
        #     count_game[id] += 1
        shuffle_player = [list_player[shuffle[0]], list_player[shuffle[1]], list_player[shuffle[2]], list_player[shuffle[3]]]
        file_temp = [[],[],[],[]]
        # try:
        winner, file_per = one_game(shuffle_player, file_temp, file_per, all_penalty)
        count[shuffle[winner]] += 1
    return count, file_per
























