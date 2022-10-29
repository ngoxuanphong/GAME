import numpy as np
import random as rd
from numba import njit
from base.Splendor_v2.Agent_intern import *
normal_cards_infor = np.array([[0, 2, 2, 2, 0, 0, 0], [0, 2, 3, 0, 0, 0, 0], [0, 2, 1, 1, 0, 2, 1], [0, 2, 0, 1, 0, 0, 2], [0, 2, 0, 3, 1, 0, 1], [0, 2, 1, 1, 0, 1, 1], [1, 2, 0, 0, 0, 4, 0], [0, 2, 2, 1, 0, 2, 0], [0, 1, 2, 0, 2, 0, 1], [0, 1, 0, 0, 2, 2, 0], [0, 1, 1, 0, 1, 1, 1], [0, 1, 2, 0, 1, 1, 1], [0, 1, 1, 1, 3, 0, 0], [0, 1, 0, 0, 0, 2, 1], [0, 1, 0, 0, 0, 3, 0], [1, 1, 4, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 4], [0, 0, 0, 0, 0, 0, 3], [0, 0, 0, 1, 1, 1, 2], [0, 0, 0, 0, 1, 2, 2], [0, 0, 1, 0, 0, 3, 1], [0, 0, 2, 0, 0, 0, 2], [0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 2, 1, 0, 0], [0, 4, 0, 2, 2, 1, 0], [0, 4, 1, 1, 2, 1, 0], [0, 4, 0, 1, 0, 1, 3], [1, 4, 0, 0, 4, 0, 0], [0, 4, 0, 2, 0, 2, 0], [0, 4, 2, 0, 0, 1, 0], [0, 4, 1, 1, 1, 1, 0], [0, 4, 0, 3, 0, 0, 0], [0, 3, 1, 0, 2, 0, 0], [0, 3, 1, 1, 1, 0, 1], [1, 3, 0, 4, 0, 0, 0], [0, 3, 1, 2, 0, 0, 2], [0, 3, 0, 0, 3, 0, 0], [0, 3, 0, 0, 2, 0, 2], [0, 3, 3, 0, 1, 1, 0], [0, 3, 1, 2, 1, 0, 1], [1, 2, 0, 3, 0, 2, 2], [2, 2, 0, 2, 0, 1, 4], [1, 2, 3, 0, 2, 0, 3], [2, 2, 0, 5, 3, 0, 0], [2, 2, 0, 0, 5, 0, 0], [3, 2, 0, 0, 6, 0, 0], [3, 1, 0, 6, 0, 0, 0], [2, 1, 1, 0, 0, 4, 2], [2, 1, 0, 5, 0, 0, 0], [2, 1, 0, 3, 0, 0, 5], [1, 1, 0, 2, 3, 3, 0], [1, 1, 3, 2, 2, 0, 0], [3, 0, 6, 0, 0, 0, 0], [2, 0, 0, 0, 0, 5, 3], [2, 0, 0, 0, 0, 5, 0], [2, 0, 0, 4, 2, 0, 1], [1, 0, 2, 3, 0, 3, 0], [1, 0, 2, 0, 0, 3, 2], [3, 4, 0, 0, 0, 0, 6], [2, 4, 5, 0, 0, 3, 0], [2, 4, 5, 0, 0, 0, 0], [1, 4, 3, 3, 0, 0, 2], [1, 4, 2, 0, 3, 2, 0], [2, 4, 4, 0, 1, 2, 0], [1, 3, 0, 2, 2, 0, 3], [1, 3, 0, 0, 3, 2, 3], [2, 3, 2, 1, 4, 0, 0], [2, 3, 3, 0, 5, 0, 0], [2, 3, 0, 0, 0, 0, 5], [3, 3, 0, 0, 0, 6, 0], [4, 2, 0, 7, 0, 0, 0], [4, 2, 0, 6, 3, 0, 3], [5, 2, 0, 7, 3, 0, 0], [3, 2, 3, 3, 0, 3, 5], [3, 1, 3, 0, 3, 5, 3], [4, 1, 0, 0, 0, 0, 7], [5, 1, 0, 3, 0, 0, 7], [4, 1, 0, 3, 0, 3, 6], [3, 0, 0, 5, 3, 3, 3], [4, 0, 0, 0, 7, 0, 0], [5, 0, 3, 0, 7, 0, 0], [4, 0, 3, 3, 6, 0, 0], [5, 4, 0, 0, 0, 7, 3], [3, 4, 5, 3, 3, 3, 0], [4, 4, 0, 0, 0, 7, 0], [4, 4, 3, 0, 0, 6, 3], [3, 3, 3, 3, 5, 0, 3], [5, 3, 7, 0, 0, 3, 0], [4, 3, 6, 0, 3, 3, 0], [4, 3, 7, 0, 0, 0, 0]])
noble_cards_infor = np.array([[0, 4, 4, 0, 0], [3, 0, 3, 3, 0], [3, 3, 3, 0, 0], [3, 0, 0, 3, 3], [0, 3, 0, 3, 3], [4, 0, 4, 0, 0], [4, 0, 0, 4, 0], [0, 3, 3, 0, 3], [0, 4, 0, 0, 4], [0, 0, 0, 4, 4]])

#@njit()
def Reset():
    env_state = np.full(164, 0)
    env_state[:] = 0
    env_state[101:107] = np.array([7,7,7,7,7,5])
    lv1 = np.arange(40)
    lv2 = np.arange(40, 70)
    lv3 = np.arange(70, 90)
    nob = np.arange(90, 100)
    for lv in [lv1, lv2, lv3]:
        np.random.shuffle(lv)
        env_state[lv[:4]] = 5
    np.random.shuffle(nob)
    env_state[nob[:5]] = 5
    env_state[161] = lv1[4]
    env_state[162] = lv2[4]
    env_state[163] = lv3[4]

    return env_state, lv1, lv2, lv3

#@njit()
def get_list_id_card_on_lv(lv):
    if len(lv) >= 4:return lv[:4]
    else: return lv[:len(lv)]

#@njit()
def concatenate_all_lv_card(lv1, lv2, lv3):
    card_lv1 = normal_cards_infor[get_list_id_card_on_lv(lv1)]
    card_lv2 = normal_cards_infor[get_list_id_card_on_lv(lv2)]
    card_lv3 = normal_cards_infor[get_list_id_card_on_lv(lv3)]
    list_open_card = np.append(card_lv1, card_lv2)
    list_open_card = np.append(list_open_card, card_lv3)
    return list_open_card
    
#@njit()
def get_id_card_normal_in_lv(lv1, lv2, lv3):
    list_card_normal_on_board = np.append(get_list_id_card_on_lv(lv1), get_list_id_card_on_lv(lv2))
    list_card_normal_on_board = np.append(list_card_normal_on_board, get_list_id_card_on_lv(lv3))
    return list_card_normal_on_board

#@njit()
def get_player_state(env_state, lv1, lv2, lv3):
    p_id = env_state[100] % 4  #Lấy người đang chơi
    b_infor = env_state[101:107] # Lấy 6 loại nguyên liệu của bàn chơi
    p_infor = env_state[107 + 12*p_id:119 + 12*p_id]  #Lấy thông tin người đang chơi, 6 nguyên liệu trên bàn, 5 nguyên liệu mặc định, điểm

    list_open_card = concatenate_all_lv_card(lv1, lv2, lv3) #Lấy list thẻ normal đang mở trên bàn
    list_open_noble = noble_cards_infor[np.where(env_state[90:100] == 5)].flatten() #Lấy list thẻ Noble đang mở trên bàn

    state_card_normal = np.full(84, 0)
    state_card_noble = np.full(25, 0)
    state_card_normal[:len(list_open_card)] = list_open_card
    state_card_noble[:len(list_open_noble)] = list_open_noble

    list_upside_down_card = normal_cards_infor[np.where(env_state[:90] == -(p_id+1))]
    p_upside_down_card = np.full(21, 0)
    if len(list_upside_down_card) > 0:
        array_hide_card = list_upside_down_card.flatten()
        p_upside_down_card[:len(array_hide_card)] = array_hide_card
    
    st_getting = env_state[155:160] #Lấy thông tin 5 nguyên liệu đang lấy trong turn
    other_scores = [env_state[118 + 12 * id_other_player] for id_other_player in range(4) if id_other_player != p_id] #Lấy điểm của người chơi khác

    p_state = np.append(b_infor, p_infor)
    p_state = np.append(p_state, state_card_normal) #Lấy thông tin 12 thẻ đang mở ở trên bàn
    p_state = np.append(p_state, state_card_noble)
    p_state = np.append(p_state, p_upside_down_card) #Lấy thông tin 3 thẻ đang úp

    p_state = np.append(p_state, st_getting) #Lấy thông tin 5 nguyên liệu đang lấy trong turn
    p_state = np.append(p_state, other_scores) #Lấy điểm của người chơi khác
    p_state = np.append(p_state, (env_state[161:164] != 100)*1) #Lấy thông tin của các thẻ ẩn có thẻ úp, nếu có thể úp thì là 1
    p_state = np.append(p_state, len(np.where(env_state[:90] == 5)[0])) #Số lượng thẻ có thể úp trong bàn
    
    cls_game = int(close_game(env_state))
    if cls_game == 0:
        p_state = np.append(p_state, 0)
    else:
        p_state = np.append(p_state, 1)
    return p_state.astype(np.float64)

#@njit()
def check_victory(p_state):
    scores = p_state[153:156]
    owner_score = p_state[17]

    if owner_score >= 15 and max(scores) <= owner_score:
        return 1
    if p_state[160] == 0:
        return -1
    if max(scores) >= 15 and max(scores) > owner_score:
        return 0
    if owner_score < 15 and max(scores) < 15:
        return -1

#@njit
def get_remove_card_on_lv_and_add_new_card(env_state, lv,p_id, id_card_hide, type_action, card_id):
    if type_action == 2:
        env_state[lv[4]] = -(p_id+1)
        id_card_in_level = 4
    else:
        if len(lv) > 4:
            env_state[lv[4]] = 5
        id_card_in_level = np.where(lv == card_id)[0][0]
        if type_action == 1:
            env_state[card_id] = p_id+1

    lv = np.delete(lv, id_card_in_level)
    if len(lv) > 4:
        env_state[id_card_hide] = lv[4]
    else: 
        env_state[id_card_hide] = 100
    return env_state, lv

#@njit
def step(action,env_state, lv1, lv2, lv3):
    p_id = env_state[100] % 4
    cur_p = env_state[107 + 12*p_id:119 + 12*p_id]
    b_stocks = env_state[101:107]

    if action == 0:
        env_state[100] += 1 #Sang turn mới
        env_state[155:160] = [0,0,0,0,0]
    else:
        if 1 <= action and action < 16:#Mua thẻ
            if 1 <= action and action < 13:
                id_action = action - 1
                id_card_normal = get_id_card_normal_in_lv(lv1, lv2, lv3)
            else:
                id_action = action - 13
                id_card_normal = np.where(env_state[:90] == -(p_id+1))[0]
            card_id = id_card_normal[id_action]
            card_infor = normal_cards_infor[card_id]
            card_price = card_infor[-5:]
            nl_bo_ra = (card_price>cur_p[6:11]) * (card_price-cur_p[6:11])
            nl_bt = np.minimum(nl_bo_ra, cur_p[:5])
            nl_auto = np.sum(nl_bo_ra - nl_bt)
            
            # Trả nguyên liệu
            cur_p[:5] -= nl_bt
            cur_p[5] -= nl_auto
            b_stocks[:5] += nl_bt
            b_stocks[5] += nl_auto

            x_ = env_state[card_id]
            env_state[card_id] = p_id+1
            if x_ == 5: #Type_action == 1
                if card_id < 40:
                    env_state, lv1 = get_remove_card_on_lv_and_add_new_card(env_state, lv1,p_id, 161, 1,card_id)
                elif card_id >= 40 and card_id < 70:
                    env_state, lv2 = get_remove_card_on_lv_and_add_new_card(env_state, lv2,p_id, 162, 1,card_id)
                    env_state[card_id] = p_id+1
                else:
                    env_state, lv3 = get_remove_card_on_lv_and_add_new_card(env_state, lv3,p_id, 163, 1,card_id)
                    
            cur_p[6:11][card_infor[1]] += 1  #const_stock
            cur_p[11] += card_infor[0] #Score

            # Check Noble
            noble_lst = []
            nobles = [i for i in range(90,100) if env_state[:100][i]==5]
            for noble_id in nobles:
                if (noble_cards_infor[noble_id-90][-5:] <= cur_p[6:11]).all():
                    noble_lst.append(noble_id)

            for noble_id in noble_lst:
                env_state[noble_id] = p_id+1
                cur_p[11] += 3
                
            env_state[100] += 1 # Sang turn mới

        elif 16 <= action and action < 31:# Úp thẻ có trên bàn
            id_action = action - 16
            # print('Chon lay the', id_action)
            if b_stocks[5] > 0:
                b_stocks[5] -= 1
                cur_p[5] += 1
            if id_action == 12: #Úp thẻ ẩn cấp 1
                env_state, lv1 = get_remove_card_on_lv_and_add_new_card(env_state, lv1,p_id, 161, 2, 0)
            elif id_action == 13: #Úp thẻ ẩn cấp 2
                env_state, lv2 = get_remove_card_on_lv_and_add_new_card(env_state, lv2,p_id, 162, 2, 0)
            elif id_action == 14: #Úp thẻ ẩn cấp 3
                env_state, lv3 = get_remove_card_on_lv_and_add_new_card(env_state, lv3,p_id, 163, 2, 0)
            else: #úp thẻ bình thường trên bàn
                id_card_normal = get_id_card_normal_in_lv(lv1, lv2, lv3)
                card_id = id_card_normal[id_action]
                env_state[card_id] = -(p_id+1)
                if card_id < 40:
                    env_state, lv1 = get_remove_card_on_lv_and_add_new_card(env_state, lv1,p_id, 161, 3,card_id)
                elif card_id >= 40 and card_id < 70:
                    env_state, lv2 = get_remove_card_on_lv_and_add_new_card(env_state, lv2,p_id, 162, 3,card_id)
                else:
                    env_state, lv3 = get_remove_card_on_lv_and_add_new_card(env_state, lv3,p_id, 163, 3,card_id)

            if np.sum(cur_p[:6]) <= 10:
                env_state[100] += 1 # Sang turn mới

        elif 31 <= action and action < 36: #Lấy nguyên liệu
            check_phase3 = False
            taken = env_state[155:160] #Các nguyên liệu đang lấy
            id_action = action - 31     #Id loại nguyên liệu lấy trong turn
            b_stocks[id_action] -= 1   #Trừ nguyên liệu bàn chơi
            cur_p[id_action] += 1      #thêm nguyên liệu của người chơi
            taken[id_action] += 1      # thêm nguyên liệu lấy trong turn
            # print('Lấy nguyên liệu:', id_action, 'Taken:',taken)
            s_taken = np.sum(taken)

            if s_taken == 1: # Chỉ còn đúng loại nl vừa lấy nhưng sl < 3
                if b_stocks[id_action] < 3 and (np.sum(b_stocks[:5]) - b_stocks[id_action]) == 0:
                    check_phase3 = True
            elif s_taken == 2: # Lấy double, hoặc không còn nl nào khác 2 cái vừa lấy
                if np.max(taken) == 2 or (np.sum(b_stocks[:5]) - np.sum(b_stocks[np.where(taken>0)[0]])) == 0:
                    check_phase3 = True
            else: # sum(taken) = 3
                check_phase3 = True

            if check_phase3:
                if np.sum(cur_p[:6]) < 10:
                    env_state[100] += 1 # Sang turn mới
                    env_state[155:160] = [0,0,0,0,0]
            env_state[155:160] = taken

        elif 36 <= action and action < 42: #Trả nguyên liệu
            st_ = action - 36
            cur_p[st_] -= 1
            b_stocks[st_] += 1

            if np.sum(cur_p[:6]) <= 10: # Thỏa mãn điều kiện này thì sang turn mới
                env_state[100] += 1 # Sang turn mới
                env_state[155:160] = [0,0,0,0,0]

    env_state[107 + 12*p_id:119 + 12*p_id] = cur_p
    env_state[101:107] = b_stocks
    return env_state, lv1, lv2, lv3

#@njit
def close_game(env_state):
    score_arr = np.array([env_state[118 + 12*p_id] for p_id in range(4)])
    max_score = np.max(score_arr)
    if max_score >= 15 and env_state[100] % 4 == 0:
        lst_p = np.where(score_arr==max_score)[0] + 1
        if len(lst_p) == 1:
            return lst_p[0]
        else:
            lst_p_c = []
            for p_id in lst_p:
                lst_p_c.append(np.count_nonzero(env_state[:90]==p_id))
            
            lst_p_c = np.array(lst_p_c)
            min_p_c = np.min(lst_p_c)
            lst_p_win = np.where(lst_p_c==min_p_c)[0]
            if len(lst_p_win) == 1:
                return lst_p[lst_p_win[0]]
            else:
                id_max = -1
                a = -1
                for i in lst_p_win:
                    b = max(np.where(env_state[:90]==lst_p[i])[0])
                    if b > a:
                        id_max = lst_p[i]
                        a = b

                return id_max
    
    else:
        return 0



def get_id_card(card_id):
    if card_id < 40: return f'I_{card_id +1}'
    if 40 <= card_id < 70: return f'II_{card_id - 39}'
    if 70 <= card_id < 90: return f'III_{card_id - 69}'
    

def one_game_numba_2(p0, list_other, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11, print_mode):
    env, lv1, lv2, lv3 = Reset()
    list_color = ['red', 'blue', 'green', 'black', 'white', 'auto_color']
    def _print_():
        print('----------------------------------------------------------------------------------------------------')
        print('Lượt của người chơi:', env[100]%4 + 1, list_color)
        print('B_stocks:', env[101:107], 'Turn:', env[100], )
        print('Thẻ 1:', [i_+1 for i_ in get_list_id_card_on_lv(lv1)], list(lv1+1))
        print('Thẻ 2:', [i_-39 for i_ in get_list_id_card_on_lv(lv2)], list(lv2-39))
        print('Thẻ 3:', [i_-69 for i_ in get_list_id_card_on_lv(lv3)], list(lv3-69))
        print('Noble:', [i_-89 for i_ in range(90,100) if env[:100][i_] == 5])
        print('P1:', env[107:113], env[113:118], env[118], [get_id_card(i_) for i_ in range(90) if env[i_] == -1], [get_id_card(i_) for i_ in range(90) if env[i_] == 1],
            '\nP2:', env[119:125], env[125:130], env[130], [get_id_card(i_) for i_ in range(90) if env[i_] == -2], [get_id_card(i_) for i_ in range(90) if env[i_] == 2],
            '\nP3:', env[131:137], env[137:142], env[142], [get_id_card(i_) for i_ in range(90) if env[i_] == -3], [get_id_card(i_) for i_ in range(90) if env[i_] == 3],
            '\nP4:', env[143:149], env[149:154], env[154], [get_id_card(i_) for i_ in range(90) if env[i_] == -4], [get_id_card(i_) for i_ in range(90) if env[i_] == 4],)
        print('Nl đã lấy:', env[155:160],'Thẻ ẩn:', get_id_card(env[161]), get_id_card(env[162]), get_id_card(env[163]))
        print('-------')

    def _print_action_(act, p_idx):
        if act == 0:
            print(f'Người chơi {p_idx+1} kết thúc lượt:', act)
        elif act in range(1,13):
            id_action = act-1
            id_card_normal = get_id_card_normal_in_lv(lv1, lv2, lv3)
            print(f'Người chơi {p_idx+1} mở thẻ trên bàn:', get_id_card(id_card_normal[id_action]),id_action,id_card_normal)
        elif act in range(13,16):
            id_action = act-13
            id_card_normal = np.where(env[:90] == -(p_idx+1))[0]
            print(f'Người chơi {p_idx+1} chọn mở thẻ đang úp:', get_id_card(id_card_normal[id_action]),id_action,id_card_normal)
        elif act in range(16,28):
            id_action = act-16
            id_card_normal = get_id_card_normal_in_lv(lv1, lv2, lv3)
            print(f'Người chơi {p_idx+1} chọn úp thẻ trên bàn:', get_id_card(id_card_normal[id_action]), id_action,id_card_normal)
        elif act in range(28, 31):

            print(f'Người chơi {p_idx+1} chọn úp thẻ ẩn:', get_id_card(env[161 + act-28]))
        elif act in range(31, 36):
            id_action = act-31
            print(f'Người chơi {p_idx+1} lấy nguyên liệu:', list_color[id_action])
        elif act in range(36, 42):
            id_action = act-36
            print(f'Người chơi {p_idx+1} trả nguyên liệu:', list_color[id_action])


    _cc = 0
    while env[100] <= 400 and _cc <= 10000:
        idx = env[100]%4
        player_state = get_player_state(env, lv1, lv2, lv3)
        if list_other[idx] == -1:
            action = p0(player_state)
        else:
            action = get_func(player_state, list_other[idx], per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11)

        list_action = get_list_action(player_state)
        if list_action[action] != 1:
            raise Exception('Action không hợp lệ')

        if print_mode:
            _print_()
            print('________')
            _print_action_(action, idx)
        env, lv1, lv2, lv3 = step(action, env, lv1, lv2, lv3)
        if close_game(env) != 0:
            break
        
        _cc += 1

    turn = env[100]
    for idx in range(4):
        env[100] = idx
        if list_other[idx] == -1:
            p_state = get_player_state(env, lv1, lv2, lv3)
            p_state[160] = 1
            act = p0(p_state)
            if print_mode:
                print('________Đã kết thúc game__________')
                _print_()
                if np.where(list_other == -1)[0] ==  (close_game(env) - 1):
                    print('Xin chúc mừng bạn là người chiến thắng')
                else:
                    print('Người chơi đã thua, chúc bạn may mắn lần sau')

    env[100] = turn
    winner = False
    if np.where(list_other == -1)[0] ==  (close_game(env) - 1): winner = True
    else: winner = False
    return winner


#@njit()
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
    elif id == 9: return test2_Khanh_130922(player_state, per9)
    elif id == 10: return test2_Dat_130922(player_state, per10)
    else: return test2_Hieu_130922(player_state, per11)

def n_game_numba_2(p0, num_game, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11, print_mode):
    win = 0
    for _n in range(num_game):
        list_other = np.append(np.random.choice(np.arange(12), 3), -1)
        np.random.shuffle(list_other)
        winner  = one_game_numba_2(p0, list_other, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11, print_mode)
        win += winner
    return win

def intern_main(p0, n_game, print_mode = False):
    list_all_players = dict_game_for_player[game_name_]
    list_data = load_data_per2(list_all_players, 'Splendor_v2')
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
    return n_game_numba_2(p0, n_game, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11, print_mode)



