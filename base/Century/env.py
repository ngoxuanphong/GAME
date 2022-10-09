from logging import raiseExceptions
import numpy as np
import re
import numba
from numba import vectorize, jit, cuda, float64, njit, prange
import os
import time
import pandas as pd

@njit(fastmath=True, cache=True)
def all_card_in4():
    return np.array([[0, 0, 0, 0, 2, 0, 0, 0], [0, 0, 0, 0, 3, 0, 0, 0], [0, 0, 0, 0, 4, 0, 0, 0], [0, 0, 0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 2, 1, 0, 0], [0, 0, 0, 0, 0, 2, 0, 0],
                    [0, 0, 0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1], [2, 0, 0, 0, 0, 2, 0, 0], [2, 0, 0, 0, 0, 0, 1, 0], [3, 0, 0, 0, 0, 0, 0, 1], [3, 0, 0, 0, 0, 3, 0, 0], [3, 0, 0, 0, 0, 1, 1, 0], [4, 0, 0, 0, 0, 0, 2, 0],
                    [4, 0, 0, 0, 0, 0, 1, 1], [5, 0, 0, 0, 0, 0, 0, 2], [5, 0, 0, 0, 0, 0, 3, 0], [0, 1, 0, 0, 3, 0, 0, 0], [0, 2, 0, 0, 0, 0, 2, 0], [0, 2, 0, 0, 3, 0, 1, 0], [0, 2, 0, 0, 2, 0, 0, 1], [0, 3, 0, 0, 0, 0, 3, 0],
                    [0, 3, 0, 0, 0, 0, 0, 2], [0, 3, 0, 0, 1, 0, 1, 1], [0, 3, 0, 0, 2, 0, 2, 0], [0, 0, 1, 0, 4, 1, 0, 0], [0, 0, 1, 0, 1, 2, 0, 0], [0, 0, 1, 0, 0, 2, 0, 0], [0, 0, 2, 0, 2, 1, 0, 1], [0, 0, 2, 0, 0, 0, 0, 2],
                    [0, 0, 2, 0, 2, 3, 0, 0], [0, 0, 2, 0, 0, 2, 0, 1], [0, 0, 3, 0, 0, 0, 0, 3], [0, 0, 0, 1, 0, 0, 2, 0], [0, 0, 0, 1, 3, 0, 1, 0], [0, 0, 0, 1, 0, 3, 0, 0], [0, 0, 0, 1, 2, 2, 0, 0], [0, 0, 0, 1, 1, 1, 1, 0],
                    [0, 0, 0, 2, 1, 1, 3, 0], [0, 0, 0, 2, 0, 3, 2, 0], [1, 1, 0, 0, 0, 0, 0, 1], [2, 0, 1, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [-1, -1, -1, -1, -1, -1, -1, -1]])

@njit(fastmath=True, cache=True)
def all_card_point_in4():
    return np.array([[0, 0, 0, 5, 20], [0, 0, 2, 3, 18], [0, 0, 3, 2, 17], [0, 0, 0, 4, 16], [0, 2, 0, 3, 16], [0, 0, 5, 0, 15], [0, 0, 2, 2, 14],
                    [0, 3, 0, 2, 14], [2, 0, 0, 3, 14], [0, 2, 3, 0, 13], [0, 0, 4, 0, 12], [0, 2, 0, 2, 12], [0, 3, 2, 0, 12], [2, 2, 0, 0, 6], [3, 2, 0, 0, 7],
                    [2, 3, 0, 0, 8], [2, 0, 2, 0, 8], [0, 4, 0, 0, 8], [3, 0, 2, 0, 9], [2, 0, 0, 2, 10], [0, 5, 0, 0, 10], [0, 2, 2, 0, 10], [2, 0, 3, 0, 11],
                    [3, 0, 0, 2, 11], [1, 1, 1, 3, 20], [0, 2, 2, 2, 19], [1, 1, 3, 1, 18], [2, 0, 2, 2, 17], [1, 3, 1, 1, 16], [2, 2, 0, 2, 15], [3, 1, 1, 1, 14],
                    [2, 2, 2, 0, 13], [0, 2, 1, 1, 12], [1, 0, 2, 1, 12], [1, 1, 1, 1, 12], [2, 1, 0, 1, 9]])

@njit(fastmath=True, cache=True)
def amount_player():
    return 5
# khởi tạo bàn chơi
@njit(fastmath=True, cache=True)
def reset(card_in4, card_point_in4):
    start_card_player = np.concatenate((np.array([1]), np.zeros(42), np.array([1, 0])))
    start_player_0 = np.concatenate((np.array([0, 0, 3, 0, 0, 0]), start_card_player))
    start_player_1 = np.concatenate((np.array([0.1, 0, 4, 0, 0, 0]), start_card_player))
    start_player_2 = np.concatenate((np.array([0.2, 0, 4, 0, 0, 0]), start_card_player))
    start_player_3 = np.concatenate((np.array([0.3, 0, 3, 1, 0, 0]), start_card_player))
    start_player_4 = np.concatenate((np.array([0.4, 0, 3, 1, 0, 0]), start_card_player))
    list_card = np.append(np.arange(1,43), 44)
    list_card_point = np.arange(36)
    np.random.shuffle(list_card)
    np.random.shuffle(list_card_point)
    # print(list_card)
    # print(list_card_point)
    top_6_card = card_in4[list_card[:6]].flatten()
    top_5_card_point = card_point_in4[list_card_point[:5]].flatten()
    env_state = np.concatenate((start_player_0, start_player_1, start_player_2, start_player_3, start_player_4, top_6_card, np.zeros(20), top_5_card_point, list_card, list_card_point, np.array([0, -0.5, 0, 10, 10, 0, 1, 0])))
    #5 player_in4, 6card_in4,5 token free, 5 card_point_in4, list_card_shuffle, list_card_point_shuffle, [number_action, card_will_buy/card_hand_used, token need drop, silver, gold, last_action, phase, id_action]
    return env_state

@njit(fastmath=True, cache=True)
def state_to_player(env_state):
    player_action = int(env_state[-1])
    player_state = env_state[51*player_action:51*(player_action+1)]
    for idx in range(1, 5):
        id = int((player_action + idx)%5)
        all_other_player_in4 = env_state[51*id:51*(id+1)]
        all_other_player_card = np.where(all_other_player_in4[6:] == -1, 1, 0)
        player_state = np.concatenate((player_state, all_other_player_in4[:6], all_other_player_card))
    player_state = np.concatenate((player_state, env_state[255:348], env_state[-5:-1]))
    return player_state

@njit(fastmath=True, cache=True)
def amount_action():
    return 65

def player_random(player_state, file_temp, file_per):

    list_action = get_list_action(player_state)
    action = int(np.random.choice(list_action))
    # print(list_action)
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



def action_player(env_state,list_player,file_temp,file_per):
    current_player = int(env_state[-1])
    player_state = state_to_player(env_state)
    played_move,file_temp[current_player],file_per = list_player[current_player](player_state,file_temp[current_player],file_per)
    if get_list_action(player_state)[played_move] != 1:
        raise Exception('bot dua ra action khong hop le')
    return played_move,file_temp,file_per

@njit(fastmath=True, cache=True)
def get_list_action_old(player_state_origin):
    player_state = player_state_origin.copy()
    phase_env = int(player_state[-1])
    player_state_own = player_state[:51]
    '''
        Quy ước phase: 
        phase1: chọn mua thẻ (6 thẻ top và 5 thẻ point) hay đánh thẻ (thẻ trên tay) hay nghỉ ngơi (11 action mua, 45 action đánh, 1 action nghỉ)
        phase2: nếu mua thẻ top, chọn token để vào các thẻ trước thẻ mình mua (4 action)
        phase3: nếu đánh thẻ, chọn xem có thực hiện action của thẻ tiếp ko (2action, 1 cái là ko, 1 cái trùng vs action dùng thẻ)
        phase4: trả token dư thừa (4 action) (trùng phase 2)
        phase5: chọn tài nguyên nâng cấp    
    '''
    player_token = player_state_own[2:6]
    if phase_env == 1:
        #chọn mua thẻ (6 thẻ top và 5 thẻ point) hay đánh thẻ (thẻ trên tay) hay nghỉ ngơi (11 action mua, 45 action đánh, 1 action nghỉ)
        list_action = np.array([0]) #mặc định 1 action nghỉ ngơi
        #check mua 6 thẻ top
        number_token = np.sum(player_token)
        card_on_board = player_state[255:303]
        for act in range(6):
            if act <= number_token and np.sum(card_on_board[8*act:8*(act+1)]) >= 0:
                list_action = np.append(list_action, act+1)
            else:
                break
        #check mua 5 thẻ point
        all_card_point = player_state[323:348]
        for id in range(5):
            card_in4 = all_card_point[5*id:5*(id+1)][:4]
            if np.sum(card_in4 > player_token) == 0:
                list_action = np.append(list_action, id+7)
        #check đánh thẻ trên tay
        data = all_card_in4()
        for card_hand in range(6, 49):
            if player_state[card_hand] == 1:
                give = data[card_hand-6][:4]
                if np.sum(give > player_token) == 0:
                    list_action = np.append(list_action, card_hand+6)

        if player_state[49] == 1:
            if np.sum(player_token[:3] > 0) != 0:
                list_action = np.append(list_action, 55)
        if player_state[50] == 1:
            if np.sum(player_token[:3] > 0) != 0:
                list_action = np.append(list_action, 56)
        return list_action

    elif phase_env == 2:
        #nếu mua thẻ top và cần bỏ token, chọn token để vào các thẻ trước thẻ mình mua (4 action)
        list_action = np.where(player_token > 0)[0]+57
        return list_action

    elif phase_env == 3:
        #nếu đánh thẻ, chọn xem có thực hiện action của thẻ tiếp ko (2action, 1 cái là ko, 1 cái trùng vs action dùng thẻ)
        last_action = int(player_state[-2])     #CẬP NHẬT 13/8 từ player_state[-3] thành player_state[-2]
        list_action = np.array([61, last_action])
        return list_action
    
    elif phase_env == 4:
        #trả token dư thừa (4 action) sau khi đánh thẻ hoặc mua thẻ top
        list_action = np.where(player_token > 0)[0]+57
        return list_action

    elif phase_env == 5:
        list_action = np.where(player_token[:3] > 0)[0]+62
        # if len(list_action) == 0:
        #     list_action = np.array([65])
        return list_action
    
@njit(fastmath=True, cache=True)
def get_list_action(player_state_origin):
    list_action_return = np.zeros(65)
    player_state = player_state_origin.copy()
    phase_env = int(player_state[-1])
    player_state_own = player_state[:51]
    '''
        Quy ước phase: 
        phase1: chọn mua thẻ (6 thẻ top và 5 thẻ point) hay đánh thẻ (thẻ trên tay) hay nghỉ ngơi (11 action mua, 45 action đánh, 1 action nghỉ)
        phase2: nếu mua thẻ top, chọn token để vào các thẻ trước thẻ mình mua (4 action)
        phase3: nếu đánh thẻ, chọn xem có thực hiện action của thẻ tiếp ko (2action, 1 cái là ko, 1 cái trùng vs action dùng thẻ)
        phase4: trả token dư thừa (4 action) (trùng phase 2)
        phase5: chọn tài nguyên nâng cấp    
    '''
    player_token = player_state_own[2:6]
    if phase_env == 1:
        #chọn mua thẻ (6 thẻ top và 5 thẻ point) hay đánh thẻ (thẻ trên tay) hay nghỉ ngơi (11 action mua, 45 action đánh, 1 action nghỉ)
        list_action_return[0] = 1
        #check mua 6 thẻ top
        number_token = np.sum(player_token)
        card_on_board = player_state[255:303]
        for act in range(6):
            if act <= number_token and np.sum(card_on_board[8*act:8*(act+1)]) >= 0:
                list_action_return[act+1] = 1
            else:
                break
        #check mua 5 thẻ point
        all_card_point = player_state[323:348]
        for id in range(5):
            card_in4 = all_card_point[5*id:5*(id+1)][:4]
            if np.sum(card_in4 > player_token) == 0:
                list_action_return[id+7] = 1
        #check đánh thẻ trên tay
        data = all_card_in4()
        for card_hand in range(6, 49):
            if player_state[card_hand] == 1:
                give = data[card_hand-6][:4]
                if np.sum(give > player_token) == 0:
                    list_action_return[card_hand+6] = 1

        if player_state[49] == 1:
            if np.sum(player_token[:3] > 0) != 0:
                list_action_return[55] = 1
        if player_state[50] == 1:
            if np.sum(player_token[:3] > 0) != 0:
                list_action_return[56] = 1

    elif phase_env == 2:
        #nếu mua thẻ top và cần bỏ token, chọn token để vào các thẻ trước thẻ mình mua (4 action)
        list_action = np.where(player_token > 0)[0]+57
        list_action_return[list_action] = 1

    elif phase_env == 3:
        #nếu đánh thẻ, chọn xem có thực hiện action của thẻ tiếp ko (2action, 1 cái là ko, 1 cái trùng vs action dùng thẻ)
        last_action = int(player_state[-2])     #CẬP NHẬT 13/8 từ player_state[-3] thành player_state[-2]
        list_action = np.array([61, last_action])
        list_action_return[list_action] = 1
    
    elif phase_env == 4:
        #trả token dư thừa (4 action) sau khi đánh thẻ hoặc mua thẻ top
        list_action = np.where(player_token > 0)[0]+57
        list_action_return[list_action] = 1

    elif phase_env == 5:
        list_action = np.where(player_token[:3] > 0)[0]+62
        list_action_return[list_action] = 1
        
    return list_action_return
   


@njit(fastmath=True, cache=True)
def check_victory(player_state):
    value_return = -1
    end = 0
    for id_player in range(5):
        player_in4 = player_state[51*id_player:51*id_player+2]
        if player_in4[1] == 5:
            end = 1 
            break
    if end == 0:
        return value_return
    else:
        id_winner = -0.5
        max_point = 0
        sum_token_max = 0
        for id_player in range(5):
            player_in4 = player_state[51*id_player:51*(id_player+1)]
            player_point = player_in4[0] + np.sum(player_in4[3:6])

            if int(player_point) > int(max_point):
                max_point = player_point
                id_winner = id_player
                sum_token_max = np.sum(np.multiply(player_in4[2:6], np.array([0, 1, 2, 3])))

            elif int(player_point) == int(max_point):
                sum_token = np.sum(np.multiply(player_in4[2:6], np.array([0, 1, 2, 3])))
                if sum_token > sum_token_max:
                    max_point = player_point
                    id_winner = id_player
                    sum_token_max = sum_token
                elif sum_token == sum_token_max and player_point > max_point:
                    max_point = player_point
                    id_winner = id_player
                    sum_token_max = np.sum(np.multiply(player_in4[2:6], np.array([0, 1, 2, 3])))
                else:
                    pass
            else:
                pass
        if id_winner == 0:
            return 1
        else:
            return 0

@njit()
def amount_state():
    return 352

@njit(fastmath=True, cache=True)
def check_winner(env_state):
    winner = -1
    max_point = 0
    sum_token_max = 0
    end = 0
    for id_player in range(5):
        player_number_card_point = env_state[51*id_player+1]
        if player_number_card_point == 5:
            end = 1
            break
    if end == 1:
        for id_player in range(5):
            player_in4 = env_state[51*id_player:51*(id_player+1)]
            player_point = player_in4[0] + np.sum(player_in4[3:6])
            if int(player_point) > int(max_point):
                max_point = player_point
                winner = id_player
                sum_token_max = np.sum(np.multiply(player_in4[2:6], np.array([0, 1, 2, 3])))
            elif int(player_point) == int(max_point):
                sum_token = np.sum(np.multiply(player_in4[2:6], np.array([0, 1, 2, 3])))
                if sum_token > sum_token_max:
                    max_point = player_point
                    winner = id_player
                    sum_token_max = sum_token
                elif sum_token == sum_token_max and player_point > max_point:
                    max_point = player_point
                    winner = id_player
                    sum_token_max = sum_token
            else:
                pass
        return winner
    else:
        return winner

@njit(fastmath=True, cache=True)
def system_check_end(env_state):
    for id_player in range(5):
        if env_state[51*id_player: 51*(id_player+1)][1] == 5:
            return False
    return True

'''
    Quy ước phase: 
    phase1: chọn mua thẻ (6 thẻ top và 5 thẻ point) hay đánh thẻ (thẻ trên tay) hay nghỉ ngơi (11 action mua, 45 action đánh, 1 action nghỉ)
    phase2: nếu mua thẻ top, chọn token để vào các thẻ trước thẻ mình mua (4 action)
    phase3: nếu đánh thẻ, chọn xem có thực hiện action của thẻ tiếp ko (2action, 1 cái là ko, 1 cái trùng vs action dùng thẻ)
    phase4: trả token dư thừa (4 action)
    phase5: chọn token để nâng cấp

'''

# def one_game_print_mode(list_player, file_temp, file_per, card_in4, card_point_in4):
#     env_state = reset(card_in4, card_point_in4)
#     count_turn = 0
#     while system_check_end(env_state) and count_turn < 1000:
#         action, file_temp, file_per = action_player(env_state,list_player,file_temp,file_per)    
#         print(f'Turn: {count_turn} player {int(env_state[-1])} action {action} {all_action_mean[action]}  có {np.sum(env_state[51*int(env_state[-1]):51*int(env_state[-1]+1)][2:6])} nguyên liệu và {env_state[51*int(env_state[-1]):51*int(env_state[-1]+1)][:2]} điểm')     #có {env_state[51*int(env_state[-1]):51*int(env_state[-1]+1)]}
#         env_state = step(env_state, action, card_in4, card_point_in4)
#         count_turn += 1

#     winner = check_winner(env_state)
#     for id_player in range(5):
#         env_state[-2] = 1
#         id_action = env_state[-1]
#         action, file_temp, file_per = action_player(env_state,list_player,file_temp,file_per)
#         env_state[-1] = (env_state[-1] + 1)%5
    
#     return winner, file_per

@njit(fastmath=True, cache=True)
def step(env_state, action, card_in4, card_point_in4):
    phase_env = int(env_state[-2])
    id_action = int(env_state[-1])
    player_in4 = env_state[51*id_action:51*(id_action+1)]
    if phase_env == 1:
        #nếu người chơi nghỉ
        if action == 0:
            card_hand_player = player_in4[6:51]
            card_hand_player = np.where(card_hand_player == -1, 1, card_hand_player)
            player_in4[6:51] = card_hand_player
            env_state[51*id_action:51*(id_action+1)] = player_in4
            env_state[-2] = 1
            env_state[-1] = (env_state[-1] + 1)%5
        #nếu người chơi mua thẻ trên bàn
        elif action in range(1,7):
            if action == 1:     #nếu mua thẻ đầu
                #lấy thông tin 
                list_card_player = player_in4[6:]
                list_card_board = env_state[348:391]
                idx_card_buy = action - 1
                card_buy = int(list_card_board[idx_card_buy])
                all_token_free = env_state[303:323]
                token_free = all_token_free[4*idx_card_buy:4*(idx_card_buy + 1)]
                all_token_free = np.concatenate((all_token_free[4:], np.zeros(4)))          #9/8 cập nhật giảm token free
                #cập nhật giá trị
                list_card_player[card_buy] = 1
                list_card_board[idx_card_buy:] = np.append(list_card_board[idx_card_buy+1:], -1)
                top_6_card = np.zeros((6, 8))
                for i in range(6):
                    id = list_card_board[:6][i]
                    top_6_card[i] = card_in4[int(id)]
                top_6_card = top_6_card.flatten()
                player_in4[6:] = list_card_player       #cập nhật thẻ mới mua
                player_in4[2:6] += token_free            #cập nhật token free nếu có
                env_state[51*id_action:51*(id_action+1)] = player_in4
                env_state[348:391] = list_card_board    #cập nhật danh sách thẻ trên bàn
                env_state[255:303] = top_6_card         #cập nhật 6 thẻ người chơi có thể mua
                env_state[303:323] = all_token_free         #9/8 cập nhật giảm token free
                #kiểm tra có phải trả tài nguyên ko
                if np.sum(player_in4[2:6]) > 10:
                    env_state[-6] = np.sum(player_in4[2:6]) - 10
                    env_state[-2] = 4
                else:
                    #chuyển người chơi
                    env_state[-2] = 1
                    env_state[-1] = (env_state[-1] + 1)%5

            else:     #nếu mua thẻ cần đặt token
                #lấy thông tin 
                idx_card_buy = action - 1
                #đẩy thông tin vào hệ thống
                env_state[-6] = idx_card_buy
                env_state[-7] = idx_card_buy
                #chuyển phase
                env_state[-2] = 2
            
        elif action in range(7,12):
            #lấy thông tin
            list_card_point_board = env_state[391:427]
            list_coin = env_state[-5:-3]
            idx_card_buy = action - 7
            card_buy = int(list_card_point_board[idx_card_buy])
            token_fee = card_point_in4[card_buy][:4]
            free_score = 0
            if idx_card_buy < 2:
                if idx_card_buy == 0:
                    if list_coin[-1] != 0:
                        free_score = 3
                        env_state[-4] -= 1
                    else:
                        if list_coin[0] != 0:
                            free_score = 1
                            env_state[-5] -= 1
                else:
                    if list_coin[1] != 0 and list_coin[0] != 0:
                        free_score = 1
                        env_state[-5] -= 1
            #Cập nhật giá trị
            player_in4[0] += (free_score + card_point_in4[card_buy][-1])
            player_in4[1] += 1
            player_in4[2:6] -= token_fee
            list_card_point_board[idx_card_buy:] = np.append(list_card_point_board[idx_card_buy+1:], -1)
            top_5_card_point = np.zeros((5, 5))
            for i in range(5):
                id = list_card_point_board[:5][i]
                top_5_card_point[i] = card_point_in4[int(id)]
            top_5_card_point = top_5_card_point.flatten()


            env_state[51*id_action:51*(id_action+1)] = player_in4
            env_state[391:427] = list_card_point_board
            env_state[323:348] = top_5_card_point
            #Chuyển người chơi
            env_state[-2] = 1
            env_state[-1] = (env_state[-1] + 1)%5

        elif action in range(12, 57):
            #lấy thông tin 
            card_hand_player = player_in4[6:51]
            id_card_use = action - 12
            token_fee_get = card_in4[id_card_use]
            card_hand_player[id_card_use] = -1

            if np.sum(token_fee_get) == 0:  #nếu là thẻ nâng cấp
                player_in4[6:51] = card_hand_player 
                env_state[51*id_action:51*(id_action+1)] = player_in4
                env_state[-8] = id_card_use - 41
                env_state[-2] = 5
            else:
                #Cập nhật giá trị
                player_in4[2:6] = player_in4[2:6] - token_fee_get[:4] + token_fee_get[4:]
                player_in4[6:51] = card_hand_player
                env_state[51*id_action:51*(id_action+1)] = player_in4
                #nếu thẻ được dùng nhiều lần
                if np.sum(token_fee_get[:4]) > 0 and np.sum(token_fee_get[:4] > player_in4[2:6]) == 0:
                    env_state[-7] = id_card_use     #lưu trữ thẻ dùng gần nhất
                    env_state[-3] = action          #lưu trữ action_main gần nhất
                    env_state[-2] = 3               #chuyển phase
                else:   #dùng 1 lần rồi bỏ
                    if np.sum(player_in4[2:6]) > 10:    #nếu thừa nguyên liệu thì đi lược bỏ
                        env_state[-6] = np.sum(player_in4[2:6]) - 10
                        env_state[-2] = 4
                    else:
                        env_state[-2] = 1
                        env_state[-1] = (env_state[-1] + 1)%5

    elif phase_env == 2:
        #lấy thông tin
        stay_drop = int(env_state[-6])-1
        all_token_free = env_state[303:323]
        token_drop = action - 57
        #Cập nhật thông tin
        player_in4[2:6][token_drop] -= 1
        all_token_free[4*stay_drop + token_drop] += 1
        env_state[-6] -= 1

        if env_state[-6] == 0:      #Hoàn tất đặt nguyên liệu thì lấy thẻ
            #lấy thông tin 
            list_card_player = player_in4[6:]
            list_card_board = env_state[348:391]
            idx_card_buy =  int(env_state[-7])
            card_buy = int(list_card_board[idx_card_buy])
            token_free = np.zeros(4)
            if idx_card_buy != 5:
                token_free = all_token_free[4*idx_card_buy:4*(idx_card_buy + 1)]
                all_token_free = np.concatenate((all_token_free[: 4*idx_card_buy], all_token_free[4*(idx_card_buy+1): ], np.zeros(4)))      #9/8 cập nhật giảm token free
            #cập nhật giá trị
            list_card_player[card_buy] = 1
            list_card_board[idx_card_buy:] = np.append(list_card_board[idx_card_buy+1:], -1)
            # top_6_card = card_in4[np.array(list_card_board[:6], dtype = int)].flatten()
            top_6_card = np.zeros((6, 8))
            for i in range(6):
                id = list_card_board[:6][i]
                top_6_card[i] = card_in4[int(id)]
                
            # top_6_card = np.array([card_in4[int(id)] for id in list_card_board[:6]]).flatten()
            top_6_card = top_6_card.flatten()
            player_in4[6:] = list_card_player       #cập nhật thẻ mới mua
            player_in4[2:6] = player_in4[2:6] + token_free            #cập nhật token free nếu có
            env_state[51*id_action:51*(id_action+1)] = player_in4
            env_state[303:323] = all_token_free     #cập nhật token free
            env_state[348:391] = list_card_board    #cập nhật danh sách thẻ trên bàn
            env_state[255:303] = top_6_card         #cập nhật 6 thẻ người chơi có thể mua
            env_state[303:323] = all_token_free         #9/8 cập nhật giảm token free
            #Khôi phục các giá trị lưu trữ
            env_state[-7] = -0.5
            #kiểm tra có phải trả tài nguyên ko
            if np.sum(player_in4[2:6]) > 10:
                env_state[-6] = np.sum(player_in4[2:6]) - 10
                env_state[-2] = 4
            else:
                #chuyển người chơi
                env_state[-2] = 1
                env_state[-1] = (env_state[-1] + 1)%5
        else:
            env_state[51*id_action:51*(id_action+1)] = player_in4
            env_state[303:323] = all_token_free     #cập nhật token free

    elif phase_env == 3:
        if action == 61:#nếu ko action tiếp
            env_state[-7] = -0.5     #lưu trữ thẻ dùng gần nhất
            env_state[-3] = 0        #lưu trữ action_main gần nhất
            if np.sum(player_in4[2:6]) > 10:    #nếu thừa nguyên liệu thì đi lược bỏ
                env_state[-6] = np.sum(player_in4[2:6]) - 10
                env_state[-2] = 4
            else:
                env_state[-2] = 1
                env_state[-1] = (env_state[-1] + 1)%5
        else:
            #Lấy thông tin
            id_card_use = int(env_state[-7])
            token_fee_get = card_in4[id_card_use]
            #Cập nhật thông tin
            player_in4[2:6] = player_in4[2:6] - token_fee_get[:4] + token_fee_get[4:]
            env_state[51*id_action:51*(id_action+1)] = player_in4
            if np.sum(token_fee_get[:4] > player_in4[2:6]) == 0:
                # env_state[-7] = id_card_use     #lưu trữ thẻ dùng gần nhất
                # env_state[-3] = action          #lưu trữ action_main gần nhất
                env_state[-2] = 3               #chuyển phase
            else:  
                env_state[-7] = -0.5
                env_state[-3] = 0
                if np.sum(player_in4[2:6]) > 10:    #nếu thừa nguyên liệu thì đi lược bỏ
                    env_state[-6] = np.sum(player_in4[2:6]) - 10
                    env_state[-2] = 4
                else:
                    env_state[-2] = 1
                    env_state[-1] = (env_state[-1] + 1)%5

    elif phase_env == 4:
        #lấy thông tin
        # stay_drop = env_state[-6]
        token_drop = action - 57 
        #Cập nhật thông tin
        player_in4[2:6][token_drop] -= 1
        env_state[51*id_action:51*(id_action+1)] = player_in4
        env_state[-6] -= 1
        if env_state[-6] == 0:
            env_state[-2] = 1
            env_state[-1] = (env_state[-1] + 1)%5
        else:
            return env_state

    elif phase_env == 5:
        number_use = env_state[-8]
        id_update = action - 62
        # if id_update == 3:
        #     env_state[-8] = 0
        #     env_state[-2] = 1
        #     env_state[-1] = (env_state[-1] + 1)%5
        # else:
        player_in4[2:6][id_update] -= 1
        player_in4[2:6][id_update+1] += 1
        env_state[51*id_action:51*(id_action+1)] = player_in4
        env_state[-8] -= 1
        if env_state[-8] == 0 or np.sum(player_in4[2:5] > 0) == 0:
            env_state[-2] = 1
            env_state[-1] = (env_state[-1] + 1)%5
 
    return env_state

def one_game(list_player, file_temp, file_per, card_in4, card_point_in4):
    env_state = reset(card_in4, card_point_in4)
    count_turn = 0
    while system_check_end(env_state) and count_turn < 2000:
        action, file_temp, file_per = action_player(env_state,list_player,file_temp,file_per)     
        env_state = step(env_state, action, card_in4, card_point_in4)
        count_turn += 1

    winner = check_winner(env_state)
    for id_player in range(5):
        env_state[-2] = 1
        id_action = env_state[-1]
        action, file_temp, file_per = action_player(env_state,list_player,file_temp,file_per)
        env_state[-1] = (env_state[-1] + 1)%5
    
    return winner, file_per

def normal_main(list_player, times, file_per):
    count = np.zeros(len(list_player)+1)
    card_in4 = all_card_in4()
    card_point_in4 = all_card_point_in4()
    all_id_player = np.arange(len(list_player))

    for van in range(times):
        shuffle = np.random.choice(all_id_player, 5, replace=False)
        shuffle_player = [list_player[shuffle[0]], list_player[shuffle[1]], list_player[shuffle[2]], list_player[shuffle[3]], list_player[shuffle[4]]]
        file_temp = [[0],[0],[0],[0], [0]]
        winner, file_per = one_game(shuffle_player, file_temp, file_per, card_in4, card_point_in4)
        if winner == -1:
            count[winner] += 1
        else:
            count[shuffle[winner]] += 1
    return list(count.astype(np.int64)), file_per

def action_player_2(env_state,list_player,file_temp, file_per_2):
    current_player = int(env_state[-1])
    player_state = state_to_player(env_state)
    played_move,file_temp[current_player], file_per_2[current_player] = list_player[current_player](player_state,file_temp[current_player], file_per_2[current_player])
    if get_list_action(player_state)[played_move] != 1:
        raise Exception('bot dua ra action khong hop le')
    return played_move,file_temp, file_per_2

# def one_game_2(list_player, file_temp,  card_in4, card_point_in4, file_per_2):
#     env_state = reset(card_in4, card_point_in4)
#     count_turn = 0
#     while system_check_end(env_state) and count_turn < 2000:
#         action, file_temp,  file_per_2 = action_player_2(env_state,list_player,file_temp, file_per_2)     
#         env_state = step(env_state, action, card_in4, card_point_in4)
#         count_turn += 1

#     winner = check_winner(env_state)
#     for id_player in range(5):
#         env_state[-2] = 1
#         id_action = env_state[-1]
#         action, file_temp,  file_per_2 = action_player_2(env_state,list_player,file_temp, file_per_2)
#         env_state[-1] = (env_state[-1] + 1)%5
    
#     return winner, file_per_2

# def normal_main_2(list_player, times,  per_file_2):
#     count = np.zeros(len(list_player)+1)
#     card_in4 = all_card_in4()
#     card_point_in4 = all_card_point_in4()
#     all_id_player = np.arange(len(list_player))

#     for van in range(times):
#         shuffle = np.random.choice(all_id_player, 5, replace=False)
#         shuffle_player = [list_player[shuffle[0]], list_player[shuffle[1]], list_player[shuffle[2]], list_player[shuffle[3]], list_player[shuffle[4]]]
#         file_temp = [[0],[0],[0],[0], [0]]
#         file_per_2_new = [per_file_2[shuffle[i]] for i in range(amount_player())]

#         winner,  file_per_2_new = one_game_2(shuffle_player, file_temp,  card_in4, card_point_in4, file_per_2_new)

#         list_p_id_new = [list(shuffle).index(i) for i in range(amount_player())]
#         per_file_2 = [file_per_2_new[list_p_id_new[i]] for i in range(amount_player())]

#         if winner == -1:
#             count[winner] += 1
#         else:
#             count[shuffle[winner]] += 1
        
#         # print(per_file_2)
#     return list(count.astype(np.int64)),  per_file_2















from system.mainFunc import dict_game_for_player, load_data_per2
game_name_ = 'Century'
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
def get_func(player_state, id, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10):
    if id == 0: return test2_An_270922(player_state, per0)
    elif id == 1: return test2_Dat_130922(player_state, per1)
    elif id == 2: return test2_Hieu_270922(player_state, per2)
    elif id == 3: return test2_Khanh_270922(player_state, per3)
    elif id == 4: return test2_Phong_130922(player_state, per4)
    elif id == 5: return test2_An_200922(player_state, per5)
    elif id == 6: return test2_Phong_130922(player_state, per6)
    elif id == 7: return test2_Khanh_200922(player_state, per7)
    elif id == 8: return test2_Hieu_130922(player_state, per8)
    elif id == 8: return test2_Khanh_130922(player_state, per9)
    else: return test2_Dat_130922(player_state, per10)

@njit()
def one_game_numba(p0, list_other, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10):
    card_in4 = all_card_in4()
    card_point_in4 = all_card_point_in4()
    env = reset(card_in4, card_point_in4)
    _temp_ = List()
    _temp_.append(np.array([[0]]))

    count_turn = 0 
    while system_check_end(env) and count_turn < 2000:
        idx = int(env[-1])
        player_state = state_to_player(env)
        if list_other[idx] == -1:
            action, _temp_, per_player = p0(player_state,_temp_,per_player)
        else:
            action = get_func(player_state, list_other[idx], per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10)

        if get_list_action(player_state)[action] != 1:
            raise Exception('bot dua ra action khong hop le')

        env = step(env, action, card_in4, card_point_in4)
        count_turn += 1

    for p_idx in range(5):
        env[-2] = 1
        if list_other[int(env[-1])] == -1:
            act, _temp_, per_player = p0(state_to_player(env), _temp_, per_player)
        env[-1] = (env[-1] + 1)%5

    winner = False
    if np.where(list_other == -1)[0] ==  check_winner(env): winner = True
    else: winner = False
    return winner,  per_player


@njit()
def n_game_numba(p0, num_game, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10):
    win = 0
    for _n in range(num_game):
        list_other = np.append(np.random.choice(np.arange(11), amount_player()-1), -1)
        np.random.shuffle(list_other)
        winner,per_player  = one_game_numba(p0, list_other, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10)
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
    return n_game_numba(p0, n_game, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10)




# @njit()
def one_game_numba_2(p0, list_other, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10):
    card_in4 = all_card_in4()
    card_point_in4 = all_card_point_in4()
    env = reset(card_in4, card_point_in4)
    _temp_ = List()
    _temp_.append(np.array([[0]]))

    count_turn = 0 
    while system_check_end(env) and count_turn < 2000:
        idx = int(env[-1])
        player_state = state_to_player(env)
        if list_other[idx] == -1:
            action, _temp_, per_player = p0(player_state,_temp_,per_player)
        else:
            action = get_func(player_state, list_other[idx], per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10)

        if get_list_action(player_state)[action] != 1:
            raise Exception('bot dua ra action khong hop le')

        env = step(env, action, card_in4, card_point_in4)
        count_turn += 1

    for p_idx in range(5):
        env[-2] = 1
        if list_other[int(env[-1])] == -1:
            act, _temp_, per_player = p0(state_to_player(env), _temp_, per_player)
        env[-1] = (env[-1] + 1)%5

    winner = False
    if np.where(list_other == -1)[0] ==  check_winner(env): winner = True
    else: winner = False
    return winner,  per_player


# @njit()
def n_game_numba_2(p0, num_game, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10):
    win = 0
    for _n in range(num_game):
        list_other = np.append(np.random.choice(np.arange(11), amount_player()-1), -1)
        np.random.shuffle(list_other)
        winner,per_player  = one_game_numba_2(p0, list_other, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10)
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
    return n_game_numba_2(p0, n_game, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10)
