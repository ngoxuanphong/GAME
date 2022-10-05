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
def get_list_action_old(player_state_origin):
    player_state = player_state_origin.copy()
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
        list_action = np.array([16])
        all_left_right_up = player_state[171:201]
        if np.sum(all_left_right_up[:15]) > 0:
            list_action = np.append(list_action, 17)
        if np.sum(all_left_right_up[15:]) > 0:
            list_action = np.append(list_action, 18)
        return list_action

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
def get_list_action(player_state_origin):
    player_state = player_state_origin.copy()
    list_action_return = np.zeros(83)
    phase_env = player_state[-1]
    player_state_own = player_state[:96]
    if phase_env == 1:
        # list_action = np.array([15])
        list_action_return[15] = 1
        player_card = player_state_own[-15:]
        for id in range(15):
            if player_card[id] == 0:
                continue
            else:
                list_action_return[id] = 1

    elif phase_env == 2:
        #lấy thẻ từ các chồng bài: 16-bài rút, 17-lật trái, 18-lật phải
        list_action_return[16] = 1
        all_left_right_up = player_state[171:201]
        if np.sum(all_left_right_up[:15]) > 0:
            list_action_return[17] = 1
        if np.sum(all_left_right_up[15:]) > 0:
            list_action_return[18] = 1

    elif phase_env == 3:
        #trả thẻ bỏ vào  chồng bài lật: 19: lật trái, 20: lật phải
        list_action_return[np.array([19, 20])] = 1

    elif phase_env == 4:
        #lựa chọn bỏ qua thẻ
        list_action_return[36] = 1
        player_card = player_state_own[-15:]
        for id in range(15):
            if player_card[id] == 0:
                continue
            else:
                list_action_return[id+21] = 1
        if np.sum(player_state_own[-45:-30]) == 0:
            list_action_return[36] = 0

    elif phase_env == 5:
        #chọn 1 trong 4 loại hàng chính ngạch
        list_action_return[np.array([37, 38, 39, 40])] = 1

    elif phase_env == 6:
        #41, 42, 43 lần lượt là check người ở vị trí 1,2,3 sau mình, 44 là ko check nữa
        for id in range(3):
            #kiểm tra xem đã check hay chưa bằng cách xem mặt hàng khai báo
            type_bag_other_player = player_state[95+25*id+3]
            if type_bag_other_player != 0:
                list_action_return[41+id] = 1

    elif phase_env == 7:
        #45 là ko hối lộ nữa, 46 là thêm coin
        if player_state_own[0] > 0:
            list_action_return[np.array([45, 46])] = 1
        else:
            list_action_return[45] = 1

    elif phase_env == 8:
        list_action_return[62] = 1
        player_card_done = player_state_own[-75:-60]
        for id in range(15):
            if player_card_done[id] == 0:
                continue
            else:
                list_action_return[id+47] = 1

    elif phase_env == 9:
        list_action_return[78] = 1
        player_card_bag = player_state_own[-45:-30]
        for id in range(15):
            if player_card_bag[id] == 0:
                continue
            else:
                list_action_return[id+63] = 1

    elif phase_env == 10:
        #79 là có check hàng, 80 là cho thoát
        list_action_return[np.array([79, 80])] = 1

    elif phase_env == 11:
        list_action_return[np.array([81, 82])] = 1
    
    return list_action_return


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
                print(env_state)
                raise Exception('chưa bỏ thẻ vào túi')
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
def amount_action():
    return 83

@njit(fastmath=True, cache=True)
def amount_player():
    return 4

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
    if get_list_action(player_state)[played_move] != 1:
        raise Exception('bot dua ra action khong hop le')
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
        count_top1 = len(np.where(count_type == top1)[0])
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
            count_top2 = len(np.where(count_type == top2)[0])
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
    number_win = np.where(all_player_coin_int == temp_win)[0]
    if len(number_win) == 1:
        return number_win[0]
    else:
        all_legal = np.zeros(4)
        for id in number_win:
            all_legal[id] = np.sum(all_done_card[15*id:15*(id+1)][:4])
        temp_win = np.max(all_legal)
        number_win = np.where(all_legal == temp_win)[0]
        if len(number_win) == 1:
            return number_win[0]
        else:
            all_unlegal = np.zeros(4)
            for id in number_win:
                all_unlegal[id] = np.sum(all_done_card[15*id:15*(id+1)][4:])
            temp_win = np.max(all_unlegal)
            number_win = np.where(all_unlegal == temp_win)[0]
            if len(number_win) == 1:
                return number_win[0]
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

@njit(fastmath=True, cache=True)
def step_print_mode(env_state, action, all_penalty):
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
            print('lấy thẻ chồng bài úp: ', card_get)
            if len(card_get) > 0:
                for card in card_get:
                    player_card[int(card)-1] += 1
            card_down = np.concatenate((card_down[number_card_get:], np.zeros(number_card_get)))
            env_state[384:504] = card_down
            player_in4[-15:] = player_card
            env_state[96*id_action:96*(id_action+1)] = player_in4
            if np.sum(env_state[754:769]) > 0:
                env_state[-1] += 1
            else:
                env_state[-1] = 1
                env_state[-3] = (env_state[-3] + 1)%4
                if env_state[-3] == (env_state[-2] - 1)%4:
                    env_state[-3] = (env_state[-3] + 1)%4
                    env_state[-1] = 4

        else: 
            player_card = player_in4[-15:]
            card_up = env_state[504+125*(action-17):504+125*(action-16)]
            card_get = card_up[0]
            print('lấy thẻ chồng bài lật: ', card_get)
            player_card[int(card_get)-1] += 1
            card_up = np.append(card_up[1:], 0)
            env_state[504+125*(action-17):504+125*(action-16)] = card_up
            player_in4[-15:] = player_card
            env_state[96*id_action:96*(id_action+1)] = player_in4
            if np.sum(player_card) == 6:
                if np.sum(env_state[754:769]) > 0:
                    env_state[-1] += 1
                else:
                    env_state[-1] = 1
                    env_state[-3] = (env_state[-3] + 1)%4
                    if env_state[-3] == (env_state[-2] - 1)%4:
                        env_state[-3] = (env_state[-3] + 1)%4
                        env_state[-1] = 4

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

# def one_game_print_mode(list_player, file_temp, file_per, all_penalty):
#     env_state = reset()
#     count_turn = 0
#     while not system_check_end(env_state):
#         action, file_temp, file_per = action_player(env_state,list_player,file_temp,file_per)
#         print(f'Turn: {count_turn} player {env_state[-3]} {all_action_mean[action]} {env_state[96*int(env_state[-3]):96*int(env_state[-3]+1)]} và {[env_state[0], env_state[96], env_state[192],env_state[288]]}')
#         env_state = step_print_mode(env_state, action, all_penalty)
#         count_turn += 1
#     winner = check_winner(env_state)
#     for id_player in range(4):
#         id_action = env_state[-3]
#         action, file_temp, file_per = action_player(env_state,list_player,file_temp,file_per)
#         print(f'Turn: {count_turn} player {env_state[-3]} {all_action_mean[action]} {env_state[96*int(env_state[-3]):96*int(env_state[-3]+1)]} và {[env_state[0], env_state[96], env_state[192],env_state[288]]}')
#         env_state[-3] = (env_state[-3] + 1)%4
#     return winner, file_per

# def normal_main_print_mode(list_player, times, file_per):
#     count = np.zeros(len(list_player))
#     all_penalty = np.array([2, 2, 2, 2, 4, 4, 4, 4, 3, 4, 4, 4, 4, 5, 5])
#     all_id_player = np.arange(len(list_player))
#     for van in range(times):
#         shuffle = np.random.choice(all_id_player, 4, replace=False)
#         shuffle_player = [list_player[shuffle[0]], list_player[shuffle[1]], list_player[shuffle[2]], list_player[shuffle[3]]]
#         file_temp = [[0],[0],[0],[0]]
#         # try:
#         winner, file_per = one_game_print_mode(shuffle_player, file_temp, file_per, all_penalty)
#         count[shuffle[winner]] += 1
#     return list(count.astype(np.int64)), file_per

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
        action, file_temp, file_per = action_player(env_state,list_player,file_temp,file_per)
        env_state[-3] = (env_state[-3] + 1)%4

    return winner, file_per

def normal_main(list_player, times, file_per):
    count = np.zeros(len(list_player))
    all_penalty = np.array([2, 2, 2, 2, 4, 4, 4, 4, 3, 4, 4, 4, 4, 5, 5])
    all_id_player = np.arange(len(list_player))
    for van in range(times):
        shuffle = np.random.choice(all_id_player, 4, replace=False)
        shuffle_player = [list_player[shuffle[0]], list_player[shuffle[1]], list_player[shuffle[2]], list_player[shuffle[3]]]
        file_temp = [[0],[0],[0],[0]]
        # try:
        winner, file_per = one_game(shuffle_player, file_temp, file_per, all_penalty)
        count[shuffle[winner]] += 1
    return list(count.astype(np.int64)), file_per

def action_player_2(env_state,list_player,file_temp, file_per_2):
    current_player = int(env_state[-3])
    player_state = state_to_player(env_state)
    played_move,file_temp[current_player], file_per_2[current_player] = list_player[current_player](player_state,file_temp[current_player], file_per_2[current_player])
    if get_list_action(player_state)[played_move] != 1:
        raise Exception('bot dua ra action khong hop le')
    return played_move,file_temp, file_per_2

def one_game_2(list_player, file_temp,  all_penalty, file_per_2):
    env_state = reset()
    while not system_check_end(env_state):
        action, file_temp,  file_per_2 = action_player_2(env_state,list_player,file_temp, file_per_2)
        env_state = step(env_state, action, all_penalty)
    
    winner = check_winner(env_state)
    for id_player in range(4):
        env_state[-1] = 1
        id_action = env_state[-3]
        action, file_temp,  file_per_2 = action_player_2(env_state,list_player,file_temp, file_per_2)
        env_state[-3] = (env_state[-3] + 1)%4

    return winner,  file_per_2

def normal_main_2(list_player, times, per_file_2):
    count = np.zeros(len(list_player))
    all_penalty = np.array([2, 2, 2, 2, 4, 4, 4, 4, 3, 4, 4, 4, 4, 5, 5])
    all_id_player = np.arange(len(list_player))
    for van in range(times):
        shuffle = np.random.choice(all_id_player, 4, replace=False)
        file_per_2_new = [per_file_2[shuffle[i]] for i in range(amount_player())]
        shuffle_player = [list_player[shuffle[0]], list_player[shuffle[1]], list_player[shuffle[2]], list_player[shuffle[3]]]
        file_temp = [[0],[0],[0],[0]]
        winner,  file_per_2_new = one_game_2(shuffle_player, file_temp,  all_penalty, file_per_2_new)
        list_p_id_new = [list(shuffle).index(i) for i in range(amount_player())]
        per_file_2 = [file_per_2_new[list_p_id_new[i]] for i in range(amount_player())]
        count[shuffle[winner]] += 1
    return list(count.astype(np.int64)),  per_file_2

















 
from system.mainFunc import dict_game_for_player
from main import load_data_per2
game_name_ = 'Sheriff'
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
def get_func(player_state, id, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9):
    if id == 0: return test2_Phong_130922(player_state, per0)
    elif id == 1: return test2_Hieu_270922(player_state, per1)
    elif id == 2: return test2_Khanh_270922(player_state, per2)
    elif id == 3: return test2_An_200922(player_state, per3)
    elif id == 4: return test2_Phong_130922(player_state, per4)
    elif id == 5: return test2_Dat_130922(player_state, per5)
    elif id == 6: return test2_Khanh_200922(player_state, per6)
    elif id == 7: return test2_NhatAnh_200922(player_state, per7)
    elif id == 8: return test2_Dat_130922(player_state, per8)
    else: return test2_Khanh_130922(player_state, per9)

@njit()
def one_game_numba(p0, list_other, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9):
    env = reset()
    all_penalty = np.array([2, 2, 2, 2, 4, 4, 4, 4, 3, 4, 4, 4, 4, 5, 5])
    _temp_ = List()
    _temp_.append(np.array([[0]]))

    while not system_check_end(env):
        idx = int(env[-3])
        player_state = state_to_player(env)
        if list_other[idx] == -1:
            action, _temp_, per_player = p0(player_state,_temp_,per_player)
        else:
            action = get_func(player_state, list_other[idx], per0, per1, per2, per3, per4, per5, per6, per7, per8, per9)

        if get_list_action(player_state)[action] != 1:
            raise Exception('bot dua ra action khong hop le')

        env = step(env, action, all_penalty)

    for p_idx in range(4):
        env[-1] = 1
        if list_other[int(env[-3])] == -1:
            act, _temp_, per_player = p0(state_to_player(env), _temp_, per_player)
        env[-3] = (env[-3] + 1)%4

    winner = False
    if np.where(list_other == -1)[0] ==  check_winner(env): winner = True
    else: winner = False
    return winner,  per_player


@njit()
def n_game_numba(p0, num_game, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9):
    win = 0
    for _n in range(num_game):
        list_other = np.append(np.random.choice(np.arange(10), 3), -1)
        np.random.shuffle(list_other)
        winner,per_player  = one_game_numba(p0, list_other, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9)
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
    return n_game_numba(p0, n_game, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9)