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
    return played_move,file_temp,file_per

@njit(fastmath=True, cache=True)
def get_list_action(player_state):
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

# @njit
# def phase1(env_state, action, card_in4, card_point_in4):
#     id_action = int(env_state[-1])
#     player_in4 = env_state[51*id_action:51*(id_action+1)]
#     #nếu người chơi nghỉ
#     if action == 0:
#         card_hand_player = player_in4[6:51]
#         card_hand_player = np.where(card_hand_player == -1, 1, card_hand_player)
#         player_in4[6:51] = card_hand_player
#         env_state[51*id_action:51*(id_action+1)] = player_in4
#         env_state[-2] = 1
#         env_state[-1] = (env_state[-1] + 1)%5
#     #nếu người chơi mua thẻ trên bàn
#     elif action in range(1,7):
#         if action == 1:     #nếu mua thẻ đầu
#             #lấy thông tin 
#             list_card_player = player_in4[6:]
#             list_card_board = env_state[348:391]
#             idx_card_buy = action - 1
#             card_buy = int(list_card_board[idx_card_buy])
#             all_token_free = env_state[303:323]
#             token_free = all_token_free[4*idx_card_buy:4*(idx_card_buy + 1)]
#             all_token_free = np.concatenate((all_token_free[4:], np.zeros(4)))          #9/8 cập nhật giảm token free
#             #cập nhật giá trị
#             list_card_player[card_buy] = 1
#             list_card_board[idx_card_buy:] = np.append(list_card_board[idx_card_buy+1:], -1)
#             top_6_card = np.zeros((6, 8))
#             for i in range(6):
#                 id = list_card_board[:6][i]
#                 top_6_card[i] = card_in4[int(id)]
#             top_6_card = top_6_card.flatten()
#             player_in4[6:] = list_card_player       #cập nhật thẻ mới mua
#             player_in4[2:6] += token_free            #cập nhật token free nếu có
#             env_state[51*id_action:51*(id_action+1)] = player_in4
#             env_state[348:391] = list_card_board    #cập nhật danh sách thẻ trên bàn
#             env_state[255:303] = top_6_card         #cập nhật 6 thẻ người chơi có thể mua
#             env_state[303:323] = all_token_free         #9/8 cập nhật giảm token free
#             #kiểm tra có phải trả tài nguyên ko
#             if np.sum(player_in4[2:6]) > 10:
#                 env_state[-6] = np.sum(player_in4[2:6]) - 10
#                 env_state[-2] = 4
#             else:
#                 #chuyển người chơi
#                 env_state[-2] = 1
#                 env_state[-1] = (env_state[-1] + 1)%5

#         else:     #nếu mua thẻ cần đặt token
#             #lấy thông tin 
#             idx_card_buy = action - 1
#             #đẩy thông tin vào hệ thống
#             env_state[-6] = idx_card_buy
#             env_state[-7] = idx_card_buy
#             #chuyển phase
#             env_state[-2] = 2
        
#     elif action in range(7,12):
#         #lấy thông tin
#         list_card_point_board = env_state[391:427]
#         list_coin = env_state[-5:-3]
#         idx_card_buy = action - 7
#         card_buy = int(list_card_point_board[idx_card_buy])
#         token_fee = card_point_in4[card_buy][:4]
#         free_score = 0
#         if idx_card_buy < 2:
#             if idx_card_buy == 0:
#                 if list_coin[-1] != 0:
#                     free_score = 3
#                     env_state[-4] -= 1
#                 else:
#                     if list_coin[0] != 0:
#                         free_score = 1
#                         env_state[-5] -= 1
#             else:
#                 if list_coin[1] != 0 and list_coin[0] != 0:
#                     free_score = 1
#                     env_state[-5] -= 1
#         #Cập nhật giá trị
#         # print('list_coin: ', list_coin, card_buy, card_point_in4[card_buy],idx_card_buy, free_score)
#         player_in4[0] += (free_score + card_point_in4[card_buy][-1])
#         player_in4[1] += 1
#         player_in4[2:6] -= token_fee
#         list_card_point_board[idx_card_buy:] = np.append(list_card_point_board[idx_card_buy+1:], -1)
#         # top_5_card_point = card_point_in4[np.array(list_card_point_board[:5], dtype=int)].flatten()
#         # top_5_card_point = np.array([card_point_in4[int(id)] for id in list_card_point_board[:5]]).flatten()
#         top_5_card_point = np.zeros((5, 5))
#         for i in range(5):
#             id = list_card_point_board[:5][i]
#             top_5_card_point[i] = card_point_in4[int(id)]
#         # top_6_card = np.array([card_in4[int(id)] for id in list_card_board[:6]]).flatten()
#         top_5_card_point = top_5_card_point.flatten()


#         env_state[51*id_action:51*(id_action+1)] = player_in4
#         env_state[391:427] = list_card_point_board
#         env_state[323:348] = top_5_card_point
#         #Chuyển người chơi
#         env_state[-2] = 1
#         env_state[-1] = (env_state[-1] + 1)%5

#     elif action in range(12, 57):
#         #lấy thông tin 
#         card_hand_player = player_in4[6:51]
#         id_card_use = action - 12
#         token_fee_get = card_in4[id_card_use]
#         card_hand_player[id_card_use] = -1

#         if np.sum(token_fee_get) == 0:  #nếu là thẻ nâng cấp
#             player_in4[6:51] = card_hand_player 
#             env_state[51*id_action:51*(id_action+1)] = player_in4
#             env_state[-8] = id_card_use - 41
#             env_state[-2] = 5
#         else:
#             #Cập nhật giá trị
#             player_in4[2:6] = player_in4[2:6] - token_fee_get[:4] + token_fee_get[4:]
#             player_in4[6:51] = card_hand_player
#             env_state[51*id_action:51*(id_action+1)] = player_in4
#             #nếu thẻ được dùng nhiều lần
#             if np.sum(token_fee_get[:4]) > 0 and np.sum(token_fee_get[:4] > player_in4[2:6]) == 0:
#                 env_state[-7] = id_card_use     #lưu trữ thẻ dùng gần nhất
#                 env_state[-3] = action          #lưu trữ action_main gần nhất
#                 env_state[-2] = 3               #chuyển phase
#             else:   #dùng 1 lần rồi bỏ
#                 if np.sum(player_in4[2:6]) > 10:    #nếu thừa nguyên liệu thì đi lược bỏ
#                     env_state[-6] = np.sum(player_in4[2:6]) - 10
#                     env_state[-2] = 4
#                 else:
#                     env_state[-2] = 1
#                     env_state[-1] = (env_state[-1] + 1)%5

#     return env_state

# @njit
# def phase2(env_state, action, card_in4, card_point_in4):
#     id_action = int(env_state[-1])
#     player_in4 = env_state[51*id_action:51*(id_action+1)]
#     #lấy thông tin
#     stay_drop = int(env_state[-6])-1
#     all_token_free = env_state[303:323]
#     token_drop = action - 57
#     #Cập nhật thông tin
#     player_in4[2:6][token_drop] -= 1
#     all_token_free[4*stay_drop + token_drop] += 1
#     env_state[-6] -= 1

#     if env_state[-6] == 0:      #Hoàn tất đặt nguyên liệu thì lấy thẻ
#         #lấy thông tin 
#         list_card_player = player_in4[6:]
#         list_card_board = env_state[348:391]
#         idx_card_buy =  int(env_state[-7])
#         card_buy = int(list_card_board[idx_card_buy])
#         token_free = np.zeros(4)
#         if idx_card_buy != 5:
#             token_free = all_token_free[4*idx_card_buy:4*(idx_card_buy + 1)]
#             all_token_free = np.concatenate((all_token_free[: 4*idx_card_buy], all_token_free[4*(idx_card_buy+1): ], np.zeros(4)))      #9/8 cập nhật giảm token free
#         #cập nhật giá trị
#         list_card_player[card_buy] = 1
#         list_card_board[idx_card_buy:] = np.append(list_card_board[idx_card_buy+1:], -1)
#         # top_6_card = card_in4[np.array(list_card_board[:6], dtype = int)].flatten()
#         top_6_card = np.zeros((6, 8))
#         for i in range(6):
#             id = list_card_board[:6][i]
#             top_6_card[i] = card_in4[int(id)]
            
#         # top_6_card = np.array([card_in4[int(id)] for id in list_card_board[:6]]).flatten()
#         top_6_card = top_6_card.flatten()
#         player_in4[6:] = list_card_player       #cập nhật thẻ mới mua
#         player_in4[2:6] = player_in4[2:6] + token_free            #cập nhật token free nếu có
#         env_state[51*id_action:51*(id_action+1)] = player_in4
#         env_state[303:323] = all_token_free     #cập nhật token free
#         env_state[348:391] = list_card_board    #cập nhật danh sách thẻ trên bàn
#         env_state[255:303] = top_6_card         #cập nhật 6 thẻ người chơi có thể mua
#         env_state[303:323] = all_token_free         #9/8 cập nhật giảm token free
#         #Khôi phục các giá trị lưu trữ
#         env_state[-7] = -0.5
#         #kiểm tra có phải trả tài nguyên ko
#         if np.sum(player_in4[2:6]) > 10:
#             env_state[-6] = np.sum(player_in4[2:6]) - 10
#             env_state[-2] = 4
#         else:
#             #chuyển người chơi
#             env_state[-2] = 1
#             env_state[-1] = (env_state[-1] + 1)%5

#     else:
#         env_state[51*id_action:51*(id_action+1)] = player_in4
#         env_state[303:323] = all_token_free     #cập nhật token free

#     return env_state

# @njit 
# def phase3(env_state, action, card_in4, card_point_in4):
#     id_action = int(env_state[-1])
#     player_in4 = env_state[51*id_action:51*(id_action+1)]
#     if action == 61:#nếu ko action tiếp
#         env_state[-7] = -0.5     #lưu trữ thẻ dùng gần nhất
#         env_state[-3] = 0        #lưu trữ action_main gần nhất
#         if np.sum(player_in4[2:6]) > 10:    #nếu thừa nguyên liệu thì đi lược bỏ
#             env_state[-6] = np.sum(player_in4[2:6]) - 10
#             env_state[-2] = 4
#         else:
#             env_state[-2] = 1
#             env_state[-1] = (env_state[-1] + 1)%5
#     else:
#         #Lấy thông tin
#         id_card_use = int(env_state[-7])
#         token_fee_get = card_in4[id_card_use]
#         #Cập nhật thông tin
#         player_in4[2:6] = player_in4[2:6] - token_fee_get[:4] + token_fee_get[4:]
#         env_state[51*id_action:51*(id_action+1)] = player_in4
#         if np.sum(token_fee_get[:4] > player_in4[2:6]) == 0:
#             # env_state[-7] = id_card_use     #lưu trữ thẻ dùng gần nhất
#             # env_state[-3] = action          #lưu trữ action_main gần nhất
#             env_state[-2] = 3               #chuyển phase
#         else:  
#             env_state[-7] = -0.5
#             env_state[-3] = 0
#             if np.sum(player_in4[2:6]) > 10:    #nếu thừa nguyên liệu thì đi lược bỏ
#                 env_state[-6] = np.sum(player_in4[2:6]) - 10
#                 env_state[-2] = 4
#             else:
#                 env_state[-2] = 1
#                 env_state[-1] = (env_state[-1] + 1)%5
#     return env_state

# @njit 
# def phase4(env_state, action, card_in4, card_point_in4):
#     id_action = int(env_state[-1])
#     player_in4 = env_state[51*id_action:51*(id_action+1)]
#     #lấy thông tin
#     # stay_drop = env_state[-6]
#     token_drop = action - 57 
#     #Cập nhật thông tin
#     player_in4[2:6][token_drop] -= 1
#     env_state[51*id_action:51*(id_action+1)] = player_in4
#     env_state[-6] -= 1
#     if env_state[-6] == 0:
#         # print('DONE trả nguyên liệu')
#         env_state[-2] = 1
#         env_state[-1] = (env_state[-1] + 1)%5
#     return env_state

# @njit 
# def phase5(env_state, action, card_in4, card_point_in4):
#     id_action = int(env_state[-1])
#     player_in4 = env_state[51*id_action:51*(id_action+1)]
#     number_use = env_state[-8]
#     id_update = action - 62
#     player_in4[2:6][id_update] -= 1
#     player_in4[2:6][id_update+1] += 1
#     env_state[51*id_action:51*(id_action+1)] = player_in4
#     env_state[-8] -= 1
#     if env_state[-8] == 0 or np.sum(player_in4[2:5] > 0) == 0:
#         env_state[-2] = 1
#         env_state[-1] = (env_state[-1] + 1)%5
#         # print('DONE nâng cấp')
#     else:
#         # print('Nâng cấp được tiếp')
#         pass
#     return env_state

# @njit
# def step(env_state, action, card_in4, card_point_in4):
#     phase_env = int(env_state[-2])
#     if phase_env == 1:
#         env_state = phase1(env_state, action, card_in4, card_point_in4)
#     elif phase_env == 2:
#         env_state = phase2(env_state, action, card_in4, card_point_in4)
#     elif phase_env == 3:
#         env_state = phase3(env_state, action, card_in4, card_point_in4)
#     elif phase_env == 4:
#         env_state = phase4(env_state, action, card_in4, card_point_in4)
#     elif phase_env == 5:
#         env_state = phase5(env_state, action, card_in4, card_point_in4)
#     return env_state

def one_game_print_mode(list_player, file_temp, file_per, card_in4, card_point_in4):
    env_state = reset(card_in4, card_point_in4)
    count_turn = 0
    while system_check_end(env_state) and count_turn < 1000:
        action, file_temp, file_per = action_player(env_state,list_player,file_temp,file_per)    
        print(f'Turn: {count_turn} player {int(env_state[-1])} action {action} {all_action_mean[action]}  có {np.sum(env_state[51*int(env_state[-1]):51*int(env_state[-1]+1)][2:6])} nguyên liệu và {env_state[51*int(env_state[-1]):51*int(env_state[-1]+1)][:2]} điểm')     #có {env_state[51*int(env_state[-1]):51*int(env_state[-1]+1)]}
        env_state = step(env_state, action, card_in4, card_point_in4)
        count_turn += 1

    winner = check_winner(env_state)
    for id_player in range(5):
        env_state[-2] = 1
        id_action = env_state[-1]
        action, file_temp, file_per = action_player(env_state,list_player,file_temp,file_per)
        env_state[-1] = (env_state[-1] + 1)%5
    
    return winner, file_per

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
    while system_check_end(env_state) and count_turn < 1000:
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
    return count, file_per

all_action_mean = list(pd.read_excel('CENTURY.xlsx')['Mean'])


# list_player = [player_random]*5
# count_all, file_per_all = normal_main(list_player, 1000, [0])