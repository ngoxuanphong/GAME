import pandas as pd
import numpy as np
import re
import numba
from numba import vectorize, jit, cuda, float64, njit, prange
import os
import time


@njit(fastmath=True, cache=True)
def reset():
    normal_card = np.full(12, 6)
    start_player = np.concatenate((np.array([0, 1, 0, 1]), np.zeros(16)))
    card_buy_in_turn = np.zeros(12)
    environment = np.concatenate((start_player, start_player, start_player, start_player, normal_card, card_buy_in_turn, np.array([-0.5, 0, 0, 0, 0, 1])))
    #[card_sell, được đi tiếp hay ko, last_dice, pick_person, id_action, phase]

    return environment

@njit(fastmath=True, cache=True)
def state_to_player(env_state):
    player_action = int(env_state[-2])
    player_state = env_state[20*player_action:20*(player_action+1)]
    for idx in range(1, 4):
        id = int((player_action + idx)%4)
        all_other_player_in4 = env_state[20*id:20*(id+1)]
        player_state = np.append(player_state, all_other_player_in4)
    player_state = np.concatenate((player_state, env_state[80:104], env_state[-4:]))
    return player_state

@njit(fastmath=True, cache=True)
def amount_action():
    return 54

@njit(fastmath=True, cache=True)
def amount_player():
    return 4

@njit()
def amount_state():
    return 108

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
    # print(list_action)
    return action, file_temp, file_per

def action_player(env_state,list_player,file_temp,file_per):
    current_player = int(env_state[-2])
    player_state = state_to_player(env_state)
    played_move,file_temp[current_player],file_per = list_player[current_player](player_state,file_temp[current_player],file_per)
    # print(get_list_action(player_state), played_move)
    if get_list_action(player_state)[played_move] != 1:
        raise Exception('bot dua ra action khong hop le')
    return played_move,file_temp,file_per

@njit(fastmath=True, cache=True)
def check_victory(player_state):
    value_return = -1
    for id_player in range(4):
        player_in4 = player_state[20*id_player:20*(id_player+1)]
        if np.sum(player_in4[-4:]) == 4:
            value_return = id_player 
            break
    if value_return == -1:
        return value_return
    else:
        if value_return == 0:
            return 1
        else:
            return 0

@njit(fastmath=True, cache=True)
def check_winner(env_state):
    winner = -1
    for id_player in range(4):
        player_in4 = env_state[20*id_player:20*(id_player+1)]
        if np.sum(player_in4[-4:]) == 4:
            return id_player
    return winner
    
@njit(fastmath=True, cache=True)
def get_list_action_old(player_state_origin):
    player_state = player_state_origin.copy()
    # player_action = int(player_state[-2])
    phase_env = player_state[-1]
    player_state_own = player_state[:20]
    '''
        Quy ước phase: 
        phase1: chọn xúc sắc để đổ
        phase2: chọn đổ lại hay k
        phase3: chọn lấy tiền của ai
        phase4: chọn người để đổi
        phase5: chọn lá bài để đổi
        phase6: chọn lá bài muốn lấy
        phase7: chọn mua thẻ
    '''
    
    if phase_env == 1:
        if player_state_own[-1] != 0:
        #chọn số xúc sắc để đổ: 1 ứng với 1 xúc sắc, 2 ứng với 2 xúc sắc
            return np.array([1, 2]) 
        else:
            return np.array([1])  

    elif phase_env == 2:
        #chọn đổ lại hay k, 0 là ko, 1 là đổ 1, 2 là đổ 2 
        return np.array([0, 1, 2])
    
    elif phase_env == 3:
        #3, 4, 5 lần lượt là lấy tiền của người ở vị trí 1,2,3 sau mình
        all_player_coin = np.array([player_state[20], player_state[40], player_state[60]])
        id_can_stole = np.where(all_player_coin > 0)[0]
        list_action = id_can_stole + 3 
        return list_action

    elif phase_env == 4:
        #6, 7, 8 lần lượt là chọn đổi thẻ với người ở vị trí 1,2,3 sau mình, 9 là ko đổi với ai
        return np.array([6, 7, 8, 9])

    elif phase_env == 5:
        #duyệt trong các thẻ đang có, có thẻ nào đổi được thì đưa vào list_action (10-21)
        list_action = np.where(player_state_own[1:13]>0)[0] + 10
        return list_action
    elif phase_env == 6:
        #duyệt trong các thẻ của người chơi mình muốn đổi (22-33)
        player_picked_card = player_state[20*int(player_state[-3]):20*(int(player_state[-3]) + 1)][1:13]
        list_action = np.where(player_picked_card>0)[0] + 22 
        return list_action 

    elif phase_env == 7:
        #chọn mua thẻ, hành động trải từ 34-53 với 53 là hành động bỏ qua ko mua thêm
        list_action = np.array([53])
        p_coin = player_state[0]
        card_board = player_state[80:92]
        card_bought = player_state[92:104]
        if p_coin > 0:
            if card_board[0] > 0 and card_bought[0] == 0:
                list_action = np.append(list_action, 34)
        if p_coin > 0:
            if card_board[1] > 0 and card_bought[1] == 0:
                list_action = np.append(list_action, 35)
        if p_coin > 0 :
            if card_board[2] > 0 and card_bought[2] == 0:
                list_action = np.append(list_action, 36)
        if p_coin > 1 :
            if card_board[3] > 0 and card_bought[3] == 0:
                list_action = np.append(list_action, 37)
        if p_coin > 1 :
            if card_board[4] > 0 and card_bought[4] == 0:
                list_action = np.append(list_action, 38)
        if p_coin > 2 :
            if card_board[5] > 0 and card_bought[5] == 0:
                list_action = np.append(list_action, 39)
        if p_coin > 4 :
            if card_board[6] > 0 and card_bought[6] == 0:
                list_action = np.append(list_action, 40)
        if p_coin > 2 :
            if card_board[7] > 0 and card_bought[7] == 0:
                list_action = np.append(list_action, 41)
        if p_coin > 5 :
            if card_board[8] > 0 and card_bought[8] == 0:
                list_action = np.append(list_action, 42)
        if p_coin > 2 :
            if card_board[9] > 0 and card_bought[9] == 0:
                list_action = np.append(list_action, 43)
        if p_coin > 2 :
            if card_board[10] > 0 and card_bought[10] == 0:
                list_action = np.append(list_action, 44)
        if p_coin > 1 :
            if card_board[11] > 0 and card_bought[11] == 0:
                list_action = np.append(list_action, 45)

        if p_coin > 5 :
            if player_state_own[13] == 0:
                list_action = np.append(list_action, 46)
        if p_coin > 6 :
            if player_state_own[14] == 0:
                list_action = np.append(list_action, 47)
        if p_coin > 7 :
            if player_state_own[15] == 0:
                list_action = np.append(list_action, 48)

        if p_coin > 3:
            if player_state_own[-1] == 0:
                list_action = np.append(list_action, 52)
        if p_coin > 9:
            if player_state_own[-2] == 0:
                list_action = np.append(list_action, 51)
        if p_coin > 15:
            if player_state_own[-3] == 0:
                list_action = np.append(list_action, 50)
        if p_coin > 21:
            if player_state_own[-4] == 0:
                list_action = np.append(list_action, 49)
        return list_action

@njit(fastmath=True, cache=True)
def get_list_action(player_state_origin):
    list_action_return = np.zeros(54)
    player_state = player_state_origin.copy()
    phase_env = player_state[-1]
    player_state_own = player_state[:20]
    '''
        Quy ước phase: 
        phase1: chọn xúc sắc để đổ
        phase2: chọn đổ lại hay k
        phase3: chọn lấy tiền của ai
        phase4: chọn người để đổi
        phase5: chọn lá bài để đổi
        phase6: chọn lá bài muốn lấy
        phase7: chọn mua thẻ
    '''
    
    if phase_env == 1:
        if player_state_own[-1] != 0:
        #chọn số xúc sắc để đổ: 1 ứng với 1 xúc sắc, 2 ứng với 2 xúc sắc
            list_action_return[np.array([1, 2])] = 1 
        else:
            list_action_return[1] = 1
        
    elif phase_env == 2:
        #chọn đổ lại hay k, 0 là ko, 1 là đổ 1, 2 là đổ 2 
        list_action_return[np.array([0, 1, 2])] = 1
    
    elif phase_env == 3:
        #3, 4, 5 lần lượt là lấy tiền của người ở vị trí 1,2,3 sau mình
        all_player_coin = np.array([player_state[20], player_state[40], player_state[60]])
        id_can_stole = np.where(all_player_coin > 0)[0]
        list_action_return[id_can_stole + 3 ] = 1

    elif phase_env == 4:
        #6, 7, 8 lần lượt là chọn đổi thẻ với người ở vị trí 1,2,3 sau mình, 9 là ko đổi với ai
        list_action_return[np.array([6, 7, 8, 9])] = 1

    elif phase_env == 5:
        #duyệt trong các thẻ đang có, có thẻ nào đổi được thì đưa vào list_action (10-21)
        list_action = np.where(player_state_own[1:13]>0)[0] + 10
        list_action_return[list_action] = 1
        # return list_action
    elif phase_env == 6:
        #duyệt trong các thẻ của người chơi mình muốn đổi (22-33)
        player_picked_card = player_state[20*int(player_state[-3]):20*(int(player_state[-3]) + 1)][1:13]
        list_action_return[np.where(player_picked_card>0)[0] + 22 ] = 1

    elif phase_env == 7:
        #chọn mua thẻ, hành động trải từ 34-53 với 53 là hành động bỏ qua ko mua thêm
        list_action_return[53] = 1
        p_coin = player_state[0]
        card_board = player_state[80:92]
        card_bought = player_state[92:104]
        if p_coin > 0:
            if card_board[0] > 0 and card_bought[0] == 0:
                list_action_return[34] = 1
        if p_coin > 0:
            if card_board[1] > 0 and card_bought[1] == 0:
                list_action_return[35] = 1
        if p_coin > 0 :
            if card_board[2] > 0 and card_bought[2] == 0:
                list_action_return[36] = 1
        if p_coin > 1 :
            if card_board[3] > 0 and card_bought[3] == 0:
                list_action_return[37] = 1
        if p_coin > 1 :
            if card_board[4] > 0 and card_bought[4] == 0:
                list_action_return[38] = 1
        if p_coin > 2 :
            if card_board[5] > 0 and card_bought[5] == 0:
                list_action_return[39] = 1
        if p_coin > 4 :
            if card_board[6] > 0 and card_bought[6] == 0:
                list_action_return[40] = 1
        if p_coin > 2 :
            if card_board[7] > 0 and card_bought[7] == 0:
                list_action_return[41] = 1
        if p_coin > 5 :
            if card_board[8] > 0 and card_bought[8] == 0:
                list_action_return[42] = 1
        if p_coin > 2 :
            if card_board[9] > 0 and card_bought[9] == 0:
                list_action_return[43] = 1
        if p_coin > 2 :
            if card_board[10] > 0 and card_bought[10] == 0:
                list_action_return[44] = 1
        if p_coin > 1 :
            if card_board[11] > 0 and card_bought[11] == 0:
                list_action_return[45] = 1

        if p_coin > 5 :
            if player_state_own[13] == 0:
                list_action_return[46] = 1
        if p_coin > 6 :
            if player_state_own[14] == 0:
                list_action_return[47] = 1
        if p_coin > 7 :
            if player_state_own[15] == 0:
                list_action_return[48] = 1

        if p_coin > 3:
            if player_state_own[-1] == 0:
                list_action_return[52] = 1
        if p_coin > 9:
            if player_state_own[-2] == 0:
                list_action_return[51] = 1
        if p_coin > 15:
            if player_state_own[-3] == 0:
                list_action_return[50] = 1
        if p_coin > 21:
            if player_state_own[-4] == 0:
                list_action_return[49] = 1

    return list_action_return


@njit(fastmath=True, cache=True)
def step(env_state, action, all_card_fee):
    phase_env = env_state[-1]
    id_action = int(env_state[-2])
    player_in4 = env_state[20*id_action:20*(id_action+1)]
    
    if phase_env == 1:
        dice1 = 0
        dice2 = 0
        if action == 1:     #nếu đổ 1 xúc sắc
            dice1 = np.random.randint(1,7)
            dice = dice1
        elif action == 2:               #nếu đổ 2 xúc sắc
            dice1 = np.random.randint(1,7)
            dice2 = np.random.randint(1,7)
            dice =  dice1 + dice2
        env_state[-4] = dice
        if player_in4[-3] != 0 and dice1 == dice2 and dice1 != 0:
            #đánh dấu là được đổ tiếp
            env_state[-5] = 1
        if player_in4[-4] > 0:
            #nếu đổ lại đc thì truyền xem có đổ lại k
            env_state[-4] = dice
            env_state[-1] = 2
        else:
            if dice == 1:
                for id in range(4):
                    env_state[20*id] += env_state[20*id+1]

            elif dice == 2:
                for id in range(4):
                    env_state[20 * id] += env_state[20 * id + 2]

                if player_in4[-2] > 0:
                    env_state[20 * id_action] += env_state[20 * id_action + 3] * 2
                else:
                    env_state[20 * id_action] += env_state[20 * id_action + 3] 

            elif dice == 3:
                next = 1
                while 0 < player_in4[0] and next <= 3:
                    id_next = (id_action - next) % 4
                    player_id = env_state[20 * id_next : 20 * (id_next + 1)]
                    coin_get = player_id[4] * (1 + int(player_id[-2] > 0))
                    delta_coin = min(coin_get, player_in4[0])
                    player_id[0] += delta_coin
                    player_in4[0] -= delta_coin
                    env_state[20*id_next] = player_id[0]
                    next += 1

                player_in4[0] += player_in4[3] * (1 + int(player_in4[-2] > 0))          #cộng tiền từ tiệm bánh

                env_state[20 * id_action] = player_in4[0]   #cập nhật tiền của người chơi

            elif dice == 4:
                if player_in4[-2] > 0:
                    env_state[20 * id_action] += env_state[20 * id_action + 5] * 4
                else:
                    env_state[20 * id_action] += env_state[20 * id_action + 5] * 3 

            elif dice == 5:
                for id in range(4):
                    env_state[20*id] += env_state[20*id+6]

            elif dice == 6:
                if player_in4[13] > 0:
                    for next in range(1,4):
                        id_next = (id_action + next) % 4
                        delta_coin = min(2, env_state[20*id_next])
                        env_state[20*id_next] -= delta_coin
                        player_in4[0] += delta_coin
                    env_state[20*id_action:20*(id_action+1)] = player_in4

                all_other_player_coin = np.zeros(3)
                for next in range(1, 4):
                    all_other_player_coin[next-1] = env_state[20*((id_action+next)%4)]


                if player_in4[14] > 0 and np.sum(all_other_player_coin) > 0:      #nếu có thẻ đài truyền hình
                    env_state[20*id_action:20*(id_action+1)] = player_in4
                    env_state[-1] = 3           #trạng thái chọn người để lấy tiền
                    return env_state
                else:
                    if player_in4[15] > 0:  #nếu có thẻ trung tâm thương mại
                        env_state[20*id_action:20*(id_action+1)] = player_in4
                        env_state[-1] = 4       #trạng thái chọn người để đổi thẻ
                        return env_state

            elif dice == 7:
                env_state[20*id_action] += env_state[20*id_action + 7]*env_state[20*id_action+2]*3

            elif dice == 8:
                env_state[20*id_action] += env_state[20*id_action + 8]*(env_state[20*id_action+6] + env_state[20*id_action+9])*3

            elif dice == 9:
                
                next = 1
                while 0 < player_in4[0] and next <= 3:
                    id_next = (id_action - next) % 4
                    player_id = env_state[20 * id_next : 20 * (id_next + 1)]
                    coin_get = player_id[10] * (2 + int(player_id[-2] > 0))
                    delta_coin = min(coin_get, player_in4[0])
                    player_id[0] += delta_coin
                    player_in4[0] -= delta_coin
                    env_state[20*id_next] = player_id[0]
                    next += 1
                for id in range(4):
                    env_state[20*id] += env_state[20*id+9]*5
                env_state[20 * id_action] = player_in4[0]   #cập nhật tiền của người chơi

            elif dice == 10:
                
                next = 1
                while 0 < player_in4[0] and next <= 3:
                    id_next = (id_action - next) % 4
                    player_id = env_state[20 * id_next : 20 * (id_next + 1)]
                    coin_get = player_id[10] * (2 + int(player_id[-2] > 0))
                    delta_coin = min(coin_get, player_in4[0])
                    player_id[0] += delta_coin
                    player_in4[0] -= delta_coin
                    env_state[20*id_next] = player_id[0]
                    next += 1
                for id in range(4):
                    env_state[20*id] += env_state[20*id+11]*3
                env_state[20 * id_action] = player_in4[0]   #cập nhật tiền của người chơi

            elif dice == 11 or dice == 12:
                env_state[20*id_action] += env_state[20*id_action + 12]*(env_state[20*id_action+1] + env_state[20*id_action+11])*3
            #sau khi cập nhật xu, cho roll tiếp nếu đạt yêu cầu, giảm biến đánh dấu xuống
            if env_state[-5] == 1:
                env_state[-1] = 1
                env_state[-5] = 0
            else:
                if env_state[-1] == 1:
                    env_state[-1] = 7
                           
    elif phase_env == 2:
        dice = 0
        dice1 = 0
        dice2 = 0
        if action == 0:
            dice = env_state[-4]
        elif action == 1:
            env_state[-5] = 0
            dice1 = np.random.randint(1,7)
            dice = dice1
        elif action == 2:
            env_state[-5] = 0
            dice1 = np.random.randint(1,7)
            dice2 = np.random.randint(1,7)
            dice =  dice1 + dice2
        
        env_state[-4] = dice
        if player_in4[-3] != 0 and dice1 == dice2 and dice1 != 0:
            #đánh dấu là được đổ tiếp
            env_state[-5] = 1
  
        if dice == 1:
            for id in range(4):
                env_state[20*id] += env_state[20*id+1]

        elif dice == 2:
            for id in range(4):
                env_state[20 * id] += env_state[20 * id + 2]

            if player_in4[-2] > 0:
                env_state[20 * id_action] += env_state[20 * id_action + 3] * 2
            else:
                env_state[20 * id_action] += env_state[20 * id_action + 3] 

        elif dice == 3:
            next = 1
            while 0 < player_in4[0] and next <= 3:
                id_next = (id_action - next) % 4
                player_id = env_state[20 * id_next : 20 * (id_next + 1)]
                coin_get = player_id[4] * (1 + int(player_id[-2] > 0))
                delta_coin = min(coin_get, player_in4[0])
                player_id[0] += delta_coin
                player_in4[0] -= delta_coin
                env_state[20*id_next] = player_id[0]
                next += 1
            player_in4[0] += player_in4[3] * (1 + int(player_in4[-2] > 0))          #cộng tiền từ tiệm bánh
            env_state[20 * id_action] = player_in4[0]   #cập nhật tiền của người chơi

        elif dice == 4:
            if player_in4[-2] > 0:
                env_state[20 * id_action] += env_state[20 * id_action + 5] * 4
            else:
                env_state[20 * id_action] += env_state[20 * id_action + 5] * 3 

        elif dice == 5:
            for id in range(4):
                env_state[20*id] += env_state[20*id+6]

        elif dice == 6:
            if player_in4[13] > 0:
                for next in range(1,4):
                    id_next = (id_action + next) % 4
                    delta_coin = min(2, env_state[20*id_next])
                    env_state[20*id_next] -= delta_coin
                    player_in4[0] += delta_coin
                env_state[20*id_action:20*(id_action+1)] = player_in4
            
            all_other_player_coin = np.zeros(3)
            for next in range(1, 4):
                all_other_player_coin[next-1] = env_state[20*((id_action+next)%4)]

            if player_in4[14] > 0 and np.sum(all_other_player_coin) > 0:      #nếu có thẻ đài truyền hình
                env_state[20*id_action:20*(id_action+1)] = player_in4
                env_state[-1] = 3           #trạng thái chọn người để lấy tiền
                return env_state
            else:
                if player_in4[15] > 0:  #nếu có thẻ trung tâm thương mại
                    env_state[20*id_action:20*(id_action+1)] = player_in4
                    env_state[-1] = 4       #trạng thái chọn người để đổi thẻ
                else:
                    if env_state[-5] == 1:
                        env_state[-1] = 1
                        env_state[-5] = 0
                    else:
                        if env_state[-1] == 2:
                            env_state[-1] = 7
                return env_state

        elif dice == 7:
            env_state[20*id_action] += env_state[20*id_action + 7]*env_state[20*id_action+2]*3

        elif dice == 8:
            env_state[20*id_action] += env_state[20*id_action + 8]*(env_state[20*id_action+6] + env_state[20*id_action+9])*3

        elif dice == 9:
            next = 1
            while 0 < player_in4[0] and next <= 3:
                id_next = (id_action - next) % 4
                player_id = env_state[20 * id_next : 20 * (id_next + 1)]
                coin_get = player_id[10] * (2 + int(player_id[-2] > 0))
                delta_coin = min(coin_get, player_in4[0])
                player_id[0] += delta_coin
                player_in4[0] -= delta_coin
                env_state[20*id_next] = player_id[0]
                next += 1
            for id in range(4):
                env_state[20*id] += env_state[20*id+9]*5
            env_state[20 * id_action] = player_in4[0]   #cập nhật tiền của người chơi

        elif dice == 10:
            next = 1
            while 0 < player_in4[0] and next <= 3:
                id_next = (id_action - next) % 4
                player_id = env_state[20 * id_next : 20 * (id_next + 1)]
                coin_get = player_id[10] * (2 + int(player_id[-2] > 0))
                delta_coin = min(coin_get, player_in4[0])
                player_id[0] += delta_coin
                player_in4[0] -= delta_coin
                env_state[20*id_next] = player_id[0]
                next += 1
            for id in range(4):
                env_state[20*id] += env_state[20*id+11]*3
            env_state[20 * id_action] = player_in4[0]   #cập nhật tiền của người chơi

        elif dice == 11 or dice == 12:
            env_state[20*id_action] += env_state[20*id_action + 12]*(env_state[20*id_action+1] + env_state[20*id_action+11])*3
            env_state[-1] = 4  
        #sau khi cập nhật xu, cho roll tiếp nếu đạt yêu cầu, giảm biến đánh dấu xuống
        if env_state[-5] == 1:
            env_state[-1] = 1
            env_state[-5] = 0
        else:
            if env_state[-1] == 2:
                env_state[-1] = 7

    elif phase_env == 3:
        #xử lí thẻ đài truyền hình
        id_picked = int(env_state[-2]+action-2)%4
        delta_coin = min(5, env_state[20*id_picked])
        env_state[20*id_picked] -= delta_coin
        player_in4[0] += delta_coin
        if player_in4[15] > 0:  #nếu có thẻ trung tâm thương mại
            env_state[20*id_action:20*(id_action+1)] = player_in4
            env_state[-1] = 5       #trạng thái chọn người để đổi thẻ
        else:
            if env_state[-5] == 1:
                env_state[-1] = 1
                env_state[-5] = 0
            else:
                env_state[-1] = 7

    elif phase_env == 4:
        if action == 9:
            if env_state[-5] == 1:
                env_state[-1] = 1
                env_state[-5] = 0
            else:
                env_state[-1] = 7
        else:
            id_picked = int(env_state[-2]+action-5)%4
            env_state[-3] = id_picked
            env_state[-1] = 5
    
    elif phase_env == 5:
        card_sell = action - 9
        env_state[-6] = card_sell
        env_state[-1] = 6

    elif phase_env == 6:
        card_buy = action - 21
        card_sell = int(env_state[-6])
        id_picked = int(env_state[-3])
        player_picked = env_state[20*id_picked:20*(id_picked+1)]
        player_picked[card_buy] -= 1
        player_picked[card_sell] += 1

        player_in4[card_buy] += 1
        player_in4[card_sell] -= 1
        env_state[20*id_picked:20*(id_picked+1)] = player_picked
        env_state[20*id_action:20*(id_action+1)] = player_in4
        env_state[-6] = -0.5
        if env_state[-5] == 1:
            env_state[-1] = 1
            env_state[-5] = 0
        else:
            env_state[-1] = 7

    elif phase_env == 7:
        if action == 53:
            env_state[-1] = 1
            env_state[92:104] = np.zeros(12)
            env_state[-2] = (env_state[-2] + 1)%4
        else:
            card_buy = action - 33
            player_in4[card_buy] += 1
            player_in4[0] -= all_card_fee[card_buy-1]
            env_state[20*id_action:20*(id_action+1)] = player_in4
            if card_buy < 13:
                #nếu mua thẻ trên bàn thì trừ ở bàn chơi đi
                env_state[92:104][card_buy-1] += 1
                env_state[80+card_buy-1] -= 1
            if player_in4[0] == 0:
                env_state[-1] = 1
                env_state[92:104] = np.zeros(12)
                env_state[-2] = (env_state[-2] + 1)%4

    return env_state 

@njit(fastmath=True, cache=True)
def system_check_end(env_state):
    for id_player in range(4):
        if np.sum(env_state[20*id_player: 20*(id_player+1)][-4:]) == 4:
            return False
    return True

def one_game_print_mode(list_player, file_temp, file_per, all_card_fee):
    global all_action_mean
    env_state = reset()
    count_turn = 0
    while system_check_end(env_state) and count_turn < 500:
        # player_state = state_to_player(env_state)
        action, file_temp, file_per = action_player(env_state,list_player,file_temp,file_per)
        print(f'Turn: {count_turn} player {env_state[-2]} {all_action_mean[action]} {env_state[20*int(env_state[-2]):20*int(env_state[-2]+1)]} và {[env_state[0], env_state[20], env_state[40],env_state[60]]}')
        env_state = step(env_state, action, all_card_fee)
        if action == 1 or action == 2:
            print(f'Xúc sắc ra {int(env_state[-4])}')

        count_turn += 1

    winner = check_winner(env_state)
    for id_player in range(4):
        id_action = env_state[-2]
        action, file_temp, file_per = action_player(env_state,list_player,file_temp,file_per)
        env_state[-2] = (env_state[-2] + 1)%4
    return winner, file_per

def normal_main_print_mode(list_player, times, file_per):
    count = np.zeros(len(list_player)+1)
    all_card_fee = np.array([1, 1, 1, 2, 2, 3, 5, 3, 6, 3, 3, 2, 6, 7, 8, 22, 16, 10, 4])
    all_id_player = np.arange(len(list_player))
    for van in range(times):
        shuffle = np.random.choice(all_id_player, 4, replace=False)
        shuffle_player = [list_player[shuffle[0]], list_player[shuffle[1]], list_player[shuffle[2]], list_player[shuffle[3]]]
        file_temp = [[0],[0],[0],[0]]
        winner, file_per = one_game_print_mode(shuffle_player, file_temp, file_per, all_card_fee)
        if winner == -1:
            count[winner] += 1
        else:
            count[shuffle[winner]] += 1
    return count, file_per

def one_game(list_player, file_temp, file_per, all_card_fee):
    # global all_action_mean
    env_state = reset()
    count_turn = 0
    while system_check_end(env_state) and count_turn < 500:
        action, file_temp, file_per = action_player(env_state,list_player,file_temp,file_per)
        env_state = step(env_state, action, all_card_fee)
        count_turn += 1
    winner = check_winner(env_state)
    for id_player in range(4):
        env_state[-1] = 1
        id_action = env_state[-2]
        action, file_temp, file_per = action_player(env_state,list_player,file_temp,file_per)
        env_state[-2] = (env_state[-2] + 1)%4
    return winner, file_per

def normal_main(list_player, times, file_per):
    count = np.zeros(len(list_player)+1)
    all_card_fee = np.array([1, 1, 1, 2, 2, 3, 5, 3, 6, 3, 3, 2, 6, 7, 8, 22, 16, 10, 4])
    all_id_player = np.arange(len(list_player))
    for van in range(times):
        shuffle = np.random.choice(all_id_player, 4, replace=False)
        shuffle_player = [list_player[shuffle[0]], list_player[shuffle[1]], list_player[shuffle[2]], list_player[shuffle[3]]]
        file_temp = [[0],[0],[0],[0]]
        winner, file_per = one_game(shuffle_player, file_temp, file_per, all_card_fee)
        if winner == -1:
            count[winner] += 1
        else:
            count[shuffle[winner]] += 1
    return list(count.astype(np.int64)), file_per


@njit()
def numba_one_game(p_lst_idx_shuffle, p0, p1, p2, p3, all_card_fee, per_file):
    env_state = reset()
    count_turn = 0

    temp_1_player = List()
    temp_1_player.append(np.array([[0.]]))
    temp_file = [temp_1_player]*(amount_player())

    while system_check_end(env_state) and count_turn < 500:
        p_idx = int(env_state[-2])
        p_state = state_to_player(env_state)
        if p_lst_idx_shuffle[p_idx] == 0:
            act, temp_file[p_idx], per_file = p0(p_state, temp_file[p_idx], per_file)
        elif p_lst_idx_shuffle[p_idx] == 1:
            act, temp_file[p_idx], per_file = p1(p_state, temp_file[p_idx], per_file)
        elif p_lst_idx_shuffle[p_idx] == 2:
            act, temp_file[p_idx], per_file = p2(p_state, temp_file[p_idx], per_file)
        else:
            act, temp_file[p_idx], per_file = p3(p_state, temp_file[p_idx], per_file)
        if get_list_action(p_state)[act] != 1:
            raise Exception('bot dua ra action khong hop le')
        env_state = step(env_state, act, all_card_fee)
        count_turn += 1

    winner = check_winner(env_state)
    for id_player in range(4):
        env_state[-1] = 1
        id_action = env_state[-2]
        p_state = state_to_player(env_state)
        p_idx = int(env_state[-2])
        if p_lst_idx_shuffle[p_idx] == 0:
            act, temp_file[p_idx], per_file = p0(p_state, temp_file[p_idx], per_file)
        elif p_lst_idx_shuffle[p_idx] == 1:
            act, temp_file[p_idx], per_file = p1(p_state, temp_file[p_idx], per_file)
        elif p_lst_idx_shuffle[p_idx] == 2:
            act, temp_file[p_idx], per_file = p2(p_state, temp_file[p_idx], per_file)
        else:
            act, temp_file[p_idx], per_file = p3(p_state, temp_file[p_idx], per_file)
    
        env_state[-2] = (env_state[-2] + 1)%4
    return winner, per_file

@njit()
def numba_main(p0, p1, p2, p3, num_game,per_file):
    count = np.zeros(amount_player()+1)
    all_card_fee = np.array([1, 1, 1, 2, 2, 3, 5, 3, 6, 3, 3, 2, 6, 7, 8, 22, 16, 10, 4])
    p_lst_idx = np.array([0,1,2,3])
    for _n in range(num_game):
        np.random.shuffle(p_lst_idx)
        winner, per_file = numba_one_game(p_lst_idx, p0, p1, p2, p3, all_card_fee, per_file )
        if winner == -1:
            count[winner] += 1
        else:
            count[p_lst_idx[winner]] += 1
    return list(count.astype(np.int64)), per_file





from system.mainFunc import dict_game_for_player, load_data_per2
game_name_ = 'MachiKoro'
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













@njit()
def get_func(player_state, id, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11):
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
    elif id == 10: return test2_Dat_130922(player_state, per10)
    else: return test2_NhatAnh_130922(player_state, per11)


@njit()
def one_game_numba(p0, list_other, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11):
    env = reset()
    all_card_fee = np.array([1, 1, 1, 2, 2, 3, 5, 3, 6, 3, 3, 2, 6, 7, 8, 22, 16, 10, 4])
    _temp_ = List()
    _temp_.append(np.array([[0]]))
    count_turn = 0
    while system_check_end(env) and count_turn < 500:
        idx = int(env[-2])
        player_state = state_to_player(env)
        if list_other[idx] == -1:
            action, _temp_, per_player = p0(player_state,_temp_,per_player)
        else:
            action = get_func(player_state, list_other[idx], per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11)

        if get_list_action(player_state)[action] != 1:
            raise Exception('bot dua ra action khong hop le')

        env = step(env, action, all_card_fee)
        count_turn += 1

    for p_idx in range(4):
        env[-1] = 1
        if list_other[int(env[-2])] == -1:
            act, _temp_, per_player = p0(state_to_player(env), _temp_, per_player)
        env[-2] = (env[-2] + 1)%4

    winner = False
    if np.where(list_other == -1)[0] ==  check_winner(env): winner = True
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
    return n_game_numba(p0, n_game, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11)




def one_game_numba_2(p0, list_other, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11):
    env = reset()
    all_card_fee = np.array([1, 1, 1, 2, 2, 3, 5, 3, 6, 3, 3, 2, 6, 7, 8, 22, 16, 10, 4])
    _temp_ = List()
    _temp_.append(np.array([[0]]))
    count_turn = 0
    while system_check_end(env) and count_turn < 500:
        idx = int(env[-2])
        player_state = state_to_player(env)
        if list_other[idx] == -1:
            action, _temp_, per_player = p0(player_state,_temp_,per_player)
        else:
            action = get_func(player_state, list_other[idx], per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11)

        if get_list_action(player_state)[action] != 1:
            raise Exception('bot dua ra action khong hop le')

        env = step(env, action, all_card_fee)
        count_turn += 1

    for p_idx in range(4):
        env[-1] = 1
        if list_other[int(env[-2])] == -1:
            act, _temp_, per_player = p0(state_to_player(env), _temp_, per_player)
        env[-2] = (env[-2] + 1)%4

    winner = False
    if np.where(list_other == -1)[0] ==  check_winner(env): winner = True
    else: winner = False
    return winner,  per_player


# @njit()
def n_game_numba_2(p0, num_game, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11):
    win = 0
    for _n in range(num_game):
        list_other = np.append(np.random.choice(np.arange(12), 3), -1)
        np.random.shuffle(list_other)
        winner,per_player  = one_game_numba_2(p0, list_other, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11)
        win += winner
    return win, per_player

def normal_main_2(p0, per_player, n_game):
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
    return n_game_numba_2(p0, n_game, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11)