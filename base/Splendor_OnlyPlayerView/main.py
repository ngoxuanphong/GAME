import numpy as np
import random as rd
from numba import njit

normal_cards_infor = np.array([[0, 2, 2, 2, 0, 0, 0], [0, 2, 3, 0, 0, 0, 0], [0, 2, 1, 1, 0, 2, 1], [0, 2, 0, 1, 0, 0, 2], [0, 2, 0, 3, 1, 0, 1], [0, 2, 1, 1, 0, 1, 1], [1, 2, 0, 0, 0, 4, 0], [0, 2, 2, 1, 0, 2, 0], [0, 1, 2, 0, 2, 0, 1], [0, 1, 0, 0, 2, 2, 0], [0, 1, 1, 0, 1, 1, 1], [0, 1, 2, 0, 1, 1, 1], [0, 1, 1, 1, 3, 0, 0], [0, 1, 0, 0, 0, 2, 1], [0, 1, 0, 0, 0, 3, 0], [1, 1, 4, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 4], [0, 0, 0, 0, 0, 0, 3], [0, 0, 0, 1, 1, 1, 2], [0, 0, 0, 0, 1, 2, 2], [0, 0, 1, 0, 0, 3, 1], [0, 0, 2, 0, 0, 0, 2], [0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 2, 1, 0, 0], [0, 4, 0, 2, 2, 1, 0], [0, 4, 1, 1, 2, 1, 0], [0, 4, 0, 1, 0, 1, 3], [1, 4, 0, 0, 4, 0, 0], [0, 4, 0, 2, 0, 2, 0], [0, 4, 2, 0, 0, 1, 0], [0, 4, 1, 1, 1, 1, 0], [0, 4, 0, 3, 0, 0, 0], [0, 3, 1, 0, 2, 0, 0], [0, 3, 1, 1, 1, 0, 1], [1, 3, 0, 4, 0, 0, 0], [0, 3, 1, 2, 0, 0, 2], [0, 3, 0, 0, 3, 0, 0], [0, 3, 0, 0, 2, 0, 2], [0, 3, 3, 0, 1, 1, 0], [0, 3, 1, 2, 1, 0, 1], [1, 2, 0, 3, 0, 2, 2], [2, 2, 0, 2, 0, 1, 4], [1, 2, 3, 0, 2, 0, 3], [2, 2, 0, 5, 3, 0, 0], [2, 2, 0, 0, 5, 0, 0], [3, 2, 0, 0, 6, 0, 0], [3, 1, 0, 6, 0, 0, 0], [2, 1, 1, 0, 0, 4, 2], [2, 1, 0, 5, 0, 0, 0], [2, 1, 0, 3, 0, 0, 5], [1, 1, 0, 2, 3, 3, 0], [1, 1, 3, 2, 2, 0, 0], [3, 0, 6, 0, 0, 0, 0], [2, 0, 0, 0, 0, 5, 3], [2, 0, 0, 0, 0, 5, 0], [2, 0, 0, 4, 2, 0, 1], [1, 0, 2, 3, 0, 3, 0], [1, 0, 2, 0, 0, 3, 2], [3, 4, 0, 0, 0, 0, 6], [2, 4, 5, 0, 0, 3, 0], [2, 4, 5, 0, 0, 0, 0], [1, 4, 3, 3, 0, 0, 2], [1, 4, 2, 0, 3, 2, 0], [2, 4, 4, 0, 1, 2, 0], [1, 3, 0, 2, 2, 0, 3], [1, 3, 0, 0, 3, 2, 3], [2, 3, 2, 1, 4, 0, 0], [2, 3, 3, 0, 5, 0, 0], [2, 3, 0, 0, 0, 0, 5], [3, 3, 0, 0, 0, 6, 0], [4, 2, 0, 7, 0, 0, 0], [4, 2, 0, 6, 3, 0, 3], [5, 2, 0, 7, 3, 0, 0], [3, 2, 3, 3, 0, 3, 5], [3, 1, 3, 0, 3, 5, 3], [4, 1, 0, 0, 0, 0, 7], [5, 1, 0, 3, 0, 0, 7], [4, 1, 0, 3, 0, 3, 6], [3, 0, 0, 5, 3, 3, 3], [4, 0, 0, 0, 7, 0, 0], [5, 0, 3, 0, 7, 0, 0], [4, 0, 3, 3, 6, 0, 0], [5, 4, 0, 0, 0, 7, 3], [3, 4, 5, 3, 3, 3, 0], [4, 4, 0, 0, 0, 7, 0], [4, 4, 3, 0, 0, 6, 3], [3, 3, 3, 3, 5, 0, 3], [5, 3, 7, 0, 0, 3, 0], [4, 3, 6, 0, 3, 3, 0], [4, 3, 7, 0, 0, 0, 0]])
noble_cards_infor = np.array([[0, 4, 4, 0, 0], [3, 0, 3, 3, 0], [3, 3, 3, 0, 0], [3, 0, 0, 3, 3], [0, 3, 0, 3, 3], [4, 0, 4, 0, 0], [4, 0, 0, 4, 0], [0, 3, 3, 0, 3], [0, 4, 0, 0, 4], [0, 0, 0, 4, 4]])

@njit
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

@njit
def amount_action():
    return 42

@njit
def amount_player():
    return 4

@njit
def get_player_state(env_state):
    p_id = env_state[100] % 4  #Lấy người đang chơi
    b_infor = env_state[101:107] # Lấy 6 loại nguyên liệu của bàn chơi
    p_infor = env_state[107 + 12*p_id:119 + 12*p_id]  #Lấy thông tin người đang chơi, 6 nguyên liệu trên bàn, 5 nguyên liệu mặc định, điểm

    list_open_card = normal_cards_infor[np.where(env_state[:90] == 5)].flatten() #Lấy list thẻ normal đang mở trên bàn
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
    # count_card_show_board = np.array([len(np.where(env_state[:90] == 5)[0])])
    # p_state = np.concatenate((b_infor, p_infor, state_card_normal, state_card_noble, p_upside_down_card, st_getting, other_scores,count_card_show_board), axis = 0)
    return p_state

@njit
def get_list_action(p_state):
    b_stocks = p_state[:6] #Các nguyên liệu trên bàn chơi
    p_st = p_state[6:11] #Các nguyên liệu của bản thân đang có
    yellow_count = p_state[11] #Số thẻ vàng đang có
    normal_cards = p_state[18:102] #Thông tin 12 thẻ đang mở
    p_upside_down_card =  p_state[127:148] #thông tin 3 thẻ đang úp
    taken = p_state[148: 153] #các nguyên liệu đã lấy trong turn
    p_count_st = p_state[12:17] #Nguyên liệu mặc định của người chơi
    list_action = np.array([0])

    #Lấy nguyên liệu
    s_taken = np.sum(taken)
    temp_ = [i_ + 31 for i_ in range(5) if b_stocks[i_] != 0]
    if s_taken == 1:
        s_ = np.where(taken==1)[0][0]
        if b_stocks[s_] < 3: # Có thể lấy double
            if (s_+ 31) in temp_:
                temp_.remove(s_ + 31) #Xóa action đã lấy ở file temp nếu nguyên liệu không trên 4
        list_action = np.append(np.array([0]), temp_)
    elif s_taken == 2:
        lst_s_ = np.where(taken==1)[0]
        for s_ in lst_s_:
            if (s_+31) in temp_:
                temp_.remove(s_+31)
        list_action = np.append(np.array([0]), temp_)
    elif s_taken == 0:
        if len(temp_) > 0:
            list_action = np.array(temp_)
    if s_taken > 0:
        return list_action

    # Kiểm tra 15 thẻ có thể mở, action từ [1:16]
    for id_card in range(12):
        card = normal_cards[7*id_card: 7+7*id_card]
        if sum(card) > 0:
            card_need = p_st + p_count_st - card[-5:]
            if -sum(card_need[np.where(card_need < 0)]) <= yellow_count or min(card_need) >= 0: #(x*x>0)
                # print('index_card:', id_card, 'The cần:', card_need, 'thẻ',card)
                # print('tổng nguyên liệu cần', -sum(card_need[np.where(card_need < 0)]), 'Nguyên liệu vàng:', yellow_count)
                list_action = np.append(list_action, id_card+1) # check các thẻ có thể lấy
    for id_card in range(3):
        card = p_upside_down_card[7*id_card: 7+7*id_card]
        if sum(card) > 0:
            # print(p_st, p_count_st, card[-5:], card)
            card_need = p_st + p_count_st -card[-5:]
            if sum(card_need) != 0:
                if -sum(card_need[np.where(card_need < 0)]) <= yellow_count or min(card_need) >= 0:
                    list_action = np.append(list_action, id_card+13) # check các thẻ có thể lấy

    #Kiểm tra và úp thẻ, action từ [16:31]
    list_card_upside_down = []
    count_upside_down = 0
    for id_card in range(3):
        card_upside_down = p_upside_down_card[7*id_card:7+7*id_card]
        if sum(card_upside_down) > 0:
            count_upside_down += 1
        else:
            break
    if count_upside_down < 3: # Nếu chưa có đủ 3 thẻ úp thì có thể úp thêm một thẻ
        list_action_upside_down = [i+16 for i in range(0, p_state[159])]
        list_action = np.append(list_action, list_action_upside_down)
        list_card_hide = np.where(p_state[156:159] == 1)[0] + 28
        # print(list_card_hide)
        list_action = np.append(list_action, list_card_hide)
    

    #Trả nguyên liệu
    p_st_have_auto = p_state[6:12]
    sum_p_st_have_auto = sum(p_st_have_auto)
    if sum_p_st_have_auto > 10:
        # print('Nguyên liệu của bản thân', p_st_have_auto)
        list_action_return_stock = [i_+36 for i_ in range(6) if p_st_have_auto[i_] != 0]
        # print(list_action_return_stock)
        list_action = np.array(list_action_return_stock)
    if len(list_action) > 1:
        list_action = np.delete(list_action, 0)
    return list_action

@njit
def step(action,env_state, lv1, lv2, lv3):
    # if action not in list_action:
    #     # 'Action không hợp lệ'
    #     env_state[100] += 1
    #     env_state[155:160] = [0,0,0,0,0]
    #     # print('action không hợp lệ')
    #     return env_state, lv1, lv2, lv3
    # else:
    p_id = env_state[100] % 4
    cur_p = env_state[107 + 12*p_id:119 + 12*p_id]
    b_stocks = env_state[101:107]
    # print('Board:', b_stocks, 'Player:', cur_p)
    # print('id:', p_id, 'list_action:', list_action, 'action:', action)
    if action == 0:
        env_state[100] += 1 #Sang turn mới
        env_state[155:160] = [0,0,0,0,0]
    else:
        if 1 <= action and action < 16:#Mua thẻ
            if 1 <= action and action < 13:
                id_action = action - 1
                id_card_normal = np.where(env_state[:90] == 5)[0]
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
            if x_ == 5:
                if card_id < 40:
                    if len(lv1) > 4:
                        env_state[lv1[4]] = 5
                    env_state[card_id] = p_id+1
                    id_card_in_level = np.where(lv1 == card_id)[0][0]
                    lv1 = np.delete(lv1, id_card_in_level)
                    if len(lv1) > 4:
                        env_state[161] = lv1[4]
                    else: 
                        env_state[161] = 100
                elif card_id >= 40 and card_id < 70:
                    if len(lv2) > 4:
                        env_state[lv2[4]] = 5
                    env_state[card_id] = p_id+1
                    id_card_in_level = np.where(lv2 == card_id)[0][0]
                    lv2 = np.delete(lv2, id_card_in_level)
                    if len(lv2) > 4:
                        env_state[162] = lv2[4]
                    else: 
                        env_state[162] = 100
                else:
                    if len(lv3) > 4:
                        env_state[lv3[4]] = 5
                    env_state[card_id] = p_id+1
                    id_card_in_level = np.where(lv3 == card_id)[0][0]
                    lv3 = np.delete(lv3, id_card_in_level)
                    if len(lv3) > 4:
                        env_state[163] = lv3[4]
                    else: 
                        env_state[163] = 100
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
                # print(env_state[161:164], lv1, len(lv1))
                # print(list(env_state))
                env_state[lv1[4]] = -(p_id+1)
                lv1 = np.delete(lv1, 4)
                if len(lv1) > 4:
                    env_state[161] = lv1[4]
                else: 
                    env_state[161] = 100
            elif id_action == 13: #Úp thẻ ẩn cấp 2
                env_state[lv2[4]] = -(p_id+1)
                lv2 = np.delete(lv2, 4)
                if len(lv2) > 4:
                    env_state[162] = lv2[4]
                else:
                    env_state[162] = 100
            elif id_action == 14: #Úp thẻ ẩn cấp 3
                env_state[lv3[4]] = -(p_id+1)
                lv3 = np.delete(lv3, 4)
                if len(lv3) > 4:
                    env_state[163] = lv3[4]
                else:
                    env_state[163] = 100
            else: #úp thẻ bình thường trên bàn
                id_card_normal = np.where(env_state[:90] == 5)[0]
                card_id = id_card_normal[id_action]
                # print('Danh sách thẻ trên bàn:', id_card_normal, 'ID_thẻ đang lấy:', card_id)
                env_state[card_id] = -(p_id+1)
                if card_id < 40:
                    id_card_in_level = np.where(lv1 == card_id)[0][0]
                    lv1 = np.delete(lv1, id_card_in_level)
                    if len(lv1) > 4:
                        env_state[lv1[4]] = 5
                        env_state[161] = lv1[4]
                    else:
                        env_state[161] = 100
                elif card_id >= 40 and card_id < 70:
                    id_card_in_level = np.where(lv2 == card_id)[0][0]
                    lv2 = np.delete(lv2, id_card_in_level)
                    if len(lv2) > 4:
                        env_state[lv2[4]] = 5
                        env_state[162] = lv2[4]
                    else:
                        env_state[162] = 100
                else:
                    id_card_in_level = np.where(lv3 == card_id)[0][0]
                    lv3 = np.delete(lv3, id_card_in_level)
                    if len(lv3) > 4:
                        env_state[lv3[4]] = 5
                        env_state[163] = lv3[4]
                    else:
                        env_state[163] = 100
                # print('Danh sach the tren ban', id_card_normal, '   action:', id_action, '  Úp thẻ:', card_id)
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

@njit
def check_victory(p_state):
    scores = p_state[153:156]
    owner_score = p_state[17]

    if owner_score >= 15 and max(scores) <= owner_score:
        return 1
    if max(scores) >= 15 and max(scores) > owner_score:
        return 0
    if owner_score < 15 and max(scores) < 15:
        return -1

@njit
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


def one_game(list_player, per_file):
    env, lv1, lv2, lv3 = Reset()
    temp_file = [[0],[0],[0],[0]]
    _cc = 0
    while env[154] <= 400 and _cc <= 10000:
        p_idx = env[154]%4
        p_state = get_player_state(env)
        act, temp_file[p_idx], per_file = list_player[p_idx](p_state, temp_file[p_idx], per_file)
        env, lv1, lv2, lv3 = step(act, env, lv1, lv2, lv3)

        if close_game(env) != 0:
            break

        _cc += 1
    
    turn = env[100]
    for i in range(4):
        env[100] = i
        act, temp_file[i], per_file = list_player[i](get_player_state(env), temp_file[i], per_file)
    
    env[100] = turn
    return close_game(env), per_file

def normal_main(list_player, num_game=1, *per_file):
    if len(list_player) != 4:
        print('Game chỉ cho phép có đúng 4 người chơi')
        return [-1,-1,-1,-1,-1], per_file
    
    num_won = [0,0,0,0,0]
    p_lst_idx = [0,1,2,3]
    for _n in range(num_game):

        # Shuffle người chơi
        rd.shuffle(p_lst_idx)

        winner, per_file = one_game(
            [list_player[p_lst_idx[0]], list_player[p_lst_idx[1]], list_player[p_lst_idx[2]], list_player[p_lst_idx[3]]], per_file,
        )

        if winner != 0:
            num_won[p_lst_idx[winner-1]] += 1
        else:
            num_won[4] += 1

    return num_won, per_file

def one_game_print(list_player, per_file, *print_mode):
    env, lv1, lv2, lv3 = Reset()
    def _print_():
        print('----------------------------------------------------------------------------------------------------')
        print('Lượt của người chơi:', env[100]%4 + 1)
        print('Thẻ 1:', [i_ for i_ in range(40) if env[:40][i_] == 5], 'Thẻ 2:', [i_ for i_ in range(40,70) if env[:70][i_] == 5], 'Thẻ 3:', [i_ for i_ in range(70,90) if env[:90][i_] == 5], 'Thẻ noble:', [i_ for i_ in range(90,100) if env[:100][i_] == 5])
        print('B_stocks:', env[101:107], 'P1:', env[107:119], 'P2:', env[119:131], 'P3:', env[131:143], 'P4:', env[143:155])
        print('Turn:', env[100], 'Nl đã lấy:', env[155:160])
        print('Thẻ ẩn:', env[161], env[162], env[163])
        print('Thẻ đang úp:', 'P1:', [i_ for i_ in range(90) if env[i_] == -1],
        'P2:', [i_ for i_ in range(90) if env[i_] == -2],
        'P3:', [i_ for i_ in range(90) if env[i_] == -3],
        'P4:', [i_ for i_ in range(90) if env[i_] == -4])
        print('-------')


    if print_mode:
        _print_()

    temp_file = [[0],[0],[0],[0]]
    _cc = 0
    while env[154] <= 400 and _cc <= 10000:
        p_idx = env[154]%4
        p_state = get_player_state(env)
        act, temp_file[p_idx], per_file = list_player[p_idx](p_state, temp_file[p_idx], per_file)
        env, lv1, lv2, lv3 = step(act, env, lv1, lv2, lv3)
        if print_mode:
            if act == 0:
                print('Action kết thúc lượt:', act)
            elif act in range(1,13):
                print('lv1', lv1)
                print('Action mở thẻ trên bàn:', act-1)
            elif act in range(13,16):
                print('lv1', lv1)
                print('Action chọn mở thẻ đang úp:', act-13)
            elif act in range(16,28):
                print('Action chọn úp thẻ trên bàn:', act-16)
            elif act in range(28, 31):
                print('Action chọn úp thẻ ẩn:', act-28)
            elif act in range(31, 36):
                print('Action lấy nguyên liệu:', act-31)
            elif act in range(36, 42):
                print('Action trả nguyên liệu:', act-36)
            _print_()

        if close_game(env) != 0:
            break

        _cc += 1
    

    turn = env[100]
    for i in range(4):
        env[100] = i
        act, temp_file[i], per_file = list_player[i](get_player_state(env), temp_file[i], per_file)
    
    env[100] = turn
    return close_game(env), per_file

def normal_main_print(list_player, num_game=1, print_mode=False):
    per_file = [0]
    if len(list_player) != 4:
        print('Game chỉ cho phép có đúng 4 người chơi')
        return [-1,-1,-1,-1,-1], per_file
    
    num_won = [0,0,0,0,0]
    p_lst_idx = [0,1,2,3]
    for _n in range(num_game):

        # Shuffle người chơi
        rd.shuffle(p_lst_idx)
        if print_mode:
            print('Thứ tự người chơi (thứ tự này sẽ ứng với P1,P2,P3,P4):', p_lst_idx)
            print('Lưu ý: không phải người chơi index 0 là P1')

        winner, per_file = one_game_print(
            [list_player[p_lst_idx[0]], list_player[p_lst_idx[1]], list_player[p_lst_idx[2]], list_player[p_lst_idx[3]]], per_file, print_mode
        )

        if winner != 0:
            num_won[p_lst_idx[winner-1]] += 1
        else:
            num_won[4] += 1

    return num_won, per_file

