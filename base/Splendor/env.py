import numpy as np
from numba import njit
import random as rd


@njit
def generate():
    e_state = np.full(164,0)

    lv1 = np.arange(41)
    lv2 = np.arange(40,71)
    lv3 = np.arange(70,91)
    
    return e_state, lv1, lv2, lv3

@njit
def initEnv(e_state, lv1, lv2, lv3):
    e_state[:] = 0
    e_state[100:106] = [7,7,7,7,7,5]

    lv1[-1] = 4
    np.random.shuffle(lv1[:-1])
    e_state[lv1[:4]] = 5

    lv2[-1] = 4
    np.random.shuffle(lv2[:-1])
    e_state[lv2[:4]] = 5

    lv3[-1] = 4
    np.random.shuffle(lv3[:-1])
    e_state[lv3[:4]] = 5

    nob = np.arange(90,100)
    np.random.shuffle(nob)
    e_state[nob[:5]] = 5

    e_state[161:] = 1

@njit
def getAgentState(e_state, lv1, lv2, lv3):
    p_idx = e_state[154] % 4
    p_state = e_state.copy()
    if p_idx != 0:
        p_state[np.where(e_state[:100]==p_idx)[0]] = 4
        for i in range(4):
            x_ = (p_idx+i) % 4
            p_state[106+12*i:118+12*i] = e_state[106+12*x_:118+12*x_].copy()
            x_ = i+1
            y_ = (i+1-p_idx) % 4
            if x_ != p_idx:
                p_state[np.where(e_state[:100]==x_)[0]] = y_
    
    p_state[np.where(e_state[:100]<0)[0]] = 0
    p_state[np.where(e_state[:100]==-(p_idx+1))[0]] = -1
    p_state[154] = 0
    if lv1[-1] == 40:
        p_state[161] = 0
    if lv2[-1] == 30:
        p_state[162] = 0
    if lv3[-1] == 20:
        p_state[163] = 0
    cls_game = int(checkEnded(e_state))
    if cls_game == 0:
        p_state = np.append(p_state, 0)
    else:
        p_state = np.append(p_state, 1)
    return p_state.astype(np.float64)

@njit
def checkEnded(e_state):
    score_arr = e_state[np.array([117,129,141,153])]
    max_score = np.max(score_arr)
    if max_score >= 15 and e_state[160] == 0 and e_state[154] % 4 == 0:
        lst_p = np.where(score_arr==max_score)[0] + 1
        if len(lst_p) == 1:
            return lst_p[0]
        else:
            lst_p_c = []
            for p_id in lst_p:
                lst_p_c.append(np.count_nonzero(e_state[:90]==p_id))
            
            lst_p_c = np.array(lst_p_c)
            min_p_c = np.min(lst_p_c)
            lst_p_win = np.where(lst_p_c==min_p_c)[0]
            if len(lst_p_win) == 1:
                return lst_p[lst_p_win[0]]
            else:
                id_max = -1
                a = -1
                for i in lst_p_win:
                    b = max(np.where(e_state[:90]==lst_p[i])[0])
                    if b > a:
                        id_max = lst_p[i]
                        a = b

                return id_max
    
    else:
        return 0


@njit
def get_list_action_old(player_state_origin:np.int64):
    p_state = player_state_origin.copy()
    p_state = p_state.astype(np.int64)
    phase = p_state[160] # Pha
    list_action = []
    normal_cards = p_state[:90] # Trạng thái các thẻ thường
    b_stocks = p_state[100:106] # Nguyên liệu trên bàn chơi
    self_st = p_state[106:112] # Nguyên liệu của người chơi
    cards_check_buy = [i_ for i_ in range(90) if normal_cards[i_]==-1 or normal_cards[i_] == 5]

    if phase == 0: # Lựa chọn kiểu hành động
        if np.sum(b_stocks[:5]) != 0:
            list_action.append(1) # Lấy nguyên liệu
        else:
            list_action.append(0) # Bỏ lượt

        if np.count_nonzero(normal_cards==-1) < 3:
            list_action.append(2) # Úp thẻ

        for card_id in cards_check_buy:
            if check_buy_card(p_state, card_id):
                list_action.append(3) # Mua thẻ
                break
    
    elif phase == 1: # Lấy nguyên liệu
        taken = p_state[155:160]
        s_taken = np.sum(taken)
        temp_ = [i_+4 for i_ in range(5) if b_stocks[i_] != 0]
        if s_taken == 0:
            list_action += temp_
        elif s_taken == 1:
            s_ = np.where(taken==1)[0][0]
            if b_stocks[s_] >= 3: # Có thể lấy double
                list_action += temp_
            else:
                if (s_+4) in temp_:
                    temp_.remove(s_+4)

                list_action += temp_
        else:
            lst_s_ = np.where(taken==1)[0]
            for s_ in lst_s_:
                if (s_+4) in temp_:
                    temp_.remove(s_+4)
            
            list_action += temp_

        if np.sum(self_st) >= 10:
            list_action.append(0)
    
    elif phase == 2: # Úp thẻ
        temp_ = [i_+9 for i_ in range(90) if normal_cards[i_]==5]
        list_action += temp_
        if p_state[161] == 1:
            list_action.append(99)
        if p_state[162] == 1:
            list_action.append(100)
        if p_state[163] == 1:
            list_action.append(101)

    elif phase == 3: # Mua thẻ
        for card_id in cards_check_buy:
            if check_buy_card(p_state, card_id):
                list_action.append(card_id+102)
    
    else: # Pha trả nguyên liệu
        list_action += [i_+192 for i_ in range(6) if self_st[i_] != 0]
    
    return np.array(list_action)


@njit
def stepEnv(action, e_state, lv1, lv2, lv3):
    # list_action = getValidActions(getAgentState(e_state, lv1, lv2, lv3))
    # if list_action[action] != 1:
    #     '''
    #     Action không hợp lệ
    #     '''
    #     # print('Action không hợp lệ')
    #     e_state[154] += 1 # Sang turn mới
    #     e_state[160] = 0
    
    # else:
    phase = e_state[160]
    p_idx = e_state[154] % 4
    cur_p = e_state[106+12*p_idx:118+12*p_idx]
    b_stocks = e_state[100:106]

    if phase == 0: # Lựa chọn pha tiếp theo
        e_state[160] = action
        if action == 0: # Sang turn mới
            e_state[154] += 1 # Chỉnh turn
    
    elif phase == 1: # Pha lấy nguyên liệu, nguyên liệu = action - 4
        check_phase1 = False
        if action == 0:
            check_phase1 = True
        else:
            st_ = action - 4
            taken = e_state[155:160]
            taken[st_] += 1 # Thêm vào nguyên liệu đã lấy
            cur_p[st_] += 1 # Thêm nguyên liệu cho người chơi hiện tại
            b_stocks[st_] -= 1 # Trừ nguyên liệu ở bàn chơi
            # Tính toán xem pha lấy nguyên liệu còn tiếp tục hay không
            # Chuyển sang pha trả nguyên liệu hoặc sang turn mới
            s_taken = np.sum(taken)
            if s_taken == 1: # Chỉ còn đúng loại nl vừa lấy nhưng sl < 3
                if b_stocks[st_] < 3 and (np.sum(b_stocks[:5]) - b_stocks[st_]) == 0:
                    check_phase1 = True
            elif s_taken == 2: # Lấy double, hoặc không còn nl nào khác 2 cái vừa lấy
                if np.max(taken) == 2 or (np.sum(b_stocks[:5]) - np.sum(b_stocks[np.where(taken>0)[0]])) == 0:
                    check_phase1 = True
            else: # sum(taken) = 3
                check_phase1 = True
        
        if check_phase1:
            if np.sum(cur_p[:6]) > 10:
                e_state[160] = 4 # Sang pha trả nguyên liệu
            else:
                e_state[154] += 1 # Sang turn mới
                e_state[160] = 0
            
            e_state[155:160] = [0,0,0,0,0]

    elif phase == 2: # Pha úp thẻ, thẻ = action - 9, đặc biệt 90,91,92
        card_id = action - 9
        if b_stocks[5] > 0: # Check nhận nguyên liệu vàng
            cur_p[5] += 1 # Cộng nguyên liệu vàng cho người chơi
            b_stocks[5] -= 1 # Trừ nguyên liệu vàng ở bàn chơi
        
        if card_id == 90: # Úp thẻ ẩn cấp 1
            e_state[lv1[lv1[-1]]] = -(p_idx+1)
            lv1[-1] += 1
        elif card_id == 91: # Úp thẻ ẩn cấp 2
            e_state[lv2[lv2[-1]]] = -(p_idx+1)
            lv2[-1] += 1
        elif card_id == 92: # Úp thẻ ẩn cấp 3
            e_state[lv3[lv3[-1]]] = -(p_idx+1)
            lv3[-1] += 1
        else:
            e_state[card_id] = -(p_idx+1)
            if card_id < 40:
                if lv1[-1] < 40:
                    e_state[lv1[lv1[-1]]] = 5
                    lv1[-1] += 1
            elif card_id >= 40 and card_id < 70:
                if lv2[-1] < 30:
                    e_state[lv2[lv2[-1]]] = 5
                    lv2[-1] += 1
            else:
                if lv3[-1] < 20:
                    e_state[lv3[lv3[-1]]] = 5
                    lv3[-1] += 1

        # Chuyển sang pha trả nguyên liệu hoặc sang turn mới
        if np.sum(cur_p[:6]) > 10:
            e_state[160] = 4 # Sang pha trả nguyên liệu
        else:
            e_state[154] += 1 # Sang turn mới
            e_state[160] = 0
    
    elif phase == 3: # Pha mua thẻ, thẻ = action - 102
        card_id = action - 102
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

        # Nhận thẻ
        x_ = e_state[card_id]
        e_state[card_id] = p_idx+1
        if x_ == 5:
            if card_id < 40:
                if lv1[-1] < 40:
                    e_state[lv1[lv1[-1]]] = 5
                    lv1[-1] += 1
            elif card_id >= 40 and card_id < 70:
                if lv2[-1] < 30:
                    e_state[lv2[lv2[-1]]] = 5
                    lv2[-1] += 1
            else:
                if lv3[-1] < 20:
                    e_state[lv3[lv3[-1]]] = 5
                    lv3[-1] += 1

        # Score, const_stock
        cur_p[6:11][card_infor[1]] += 1
        cur_p[11] += card_infor[0]

        # Check Noble
        noble_lst = []
        nobles = [i for i in range(90,100) if e_state[:100][i]==5]
        for noble_id in nobles:
            if (noble_cards_infor[noble_id-90][-5:] <= cur_p[6:11]).all():
                noble_lst.append(noble_id)

        for noble_id in noble_lst:
            e_state[noble_id] = p_idx+1
            cur_p[11] += 3
        
        e_state[154] += 1 # Sang turn mới
        e_state[160] = 0
            
    else: # Pha trả nguyên liệu, nguyên liệu = action - 192
        st_ = action - 192
        cur_p[st_] -= 1
        b_stocks[st_] += 1

        if np.sum(cur_p[:6]) <= 10: # Thỏa mãn điều kiện này thì sang turn mới
            e_state[154] += 1 # Sang turn mới
            e_state[160] = 0


@njit
def getReward(p_state):
    score_arr = p_state[np.array([117,129,141,153])]
    max_score = np.max(score_arr)
    if p_state[164] == 0:
        return -1
    
    if max_score >= 15:
        lst_p = np.where(score_arr==max_score)[0] + 1
        if len(lst_p) == 1:
            if lst_p[0] == 1:
                return 1
            else:
                return 0
        
        else:
            lst_p_c = []
            for p_id in lst_p:
                lst_p_c.append(np.count_nonzero(p_state[:90]==p_id))
            
            lst_p_c = np.array(lst_p_c)
            min_p_c = np.min(lst_p_c)
            lst_p_win = np.where(lst_p_c==min_p_c)[0]
            if len(lst_p_win) == 1:
                if lst_p[lst_p_win[0]] == 1:
                    return 1
                else:
                    return 0
            else:
                id_max = -1
                a = -1
                for i in lst_p_win:
                    b = max(np.where(p_state[:90]==lst_p[i])[0])
                    if b > a:
                        id_max = lst_p[i]
                        a = b
                
                if id_max == 1:
                    return 1
                else:
                    return 0
    else: 
        return 0
            
def one_game(list_player, env, lv1, lv2, lv3, per_file):
    initEnv(env, lv1, lv2, lv3)
    while env[154] <= 400:
        p_idx = env[154]%4
        p_state = getAgentState(env, lv1, lv2, lv3)
        list_action = getValidActions(p_state)
        act, per_file = list_player[p_idx](p_state, per_file)
        if list_action[act] != 1:
            raise Exception('Action không hợp lệ')
        stepEnv(act, env, lv1, lv2, lv3)
        if checkEnded(env) != 0:
            break
    
    turn = env[154]
    for i in range(4):
        env[154] = i
        p_state = getAgentState(env, lv1, lv2, lv3)
        p_state[164] = 1
        act, per_file = list_player[i](p_state, per_file)
    
    env[154] = turn
    return checkEnded(env), per_file

def normal_main(list_player, num_game, per_file):
    if len(list_player) != 4:
        print('Game chỉ cho phép có đúng 4 người chơi')
        return [-1,-1,-1,-1,-1], per_file
        
    env, lv1, lv2, lv3 = generate()
    num_won = [0,0,0,0,0]
    p_lst_idx = [0,1,2,3]
    for _n in range(num_game):
        rd.shuffle(p_lst_idx)
        winner, per_file = one_game(
            [list_player[p_lst_idx[0]], list_player[p_lst_idx[1]], list_player[p_lst_idx[2]], list_player[p_lst_idx[3]]], env, lv1, lv2, lv3, per_file
        )

        if winner != 0:
            num_won[p_lst_idx[winner-1]] += 1
        else:
            num_won[4] += 1

    return num_won, per_file
        


def one_game_print(list_player, env, lv1, lv2, lv3, print_mode, per_file):
    initEnv(env, lv1, lv2, lv3)
    

    def _print_():
        print('----------------------------------------------------------------------------------------------------')
        print('Thẻ 1:', [i_ for i_ in range(40) if env[:40][i_] == 5], 'Thẻ 2:', [i_ for i_ in range(40,70) if env[:70][i_] == 5], 'Thẻ 3:', [i_ for i_ in range(70,90) if env[:90][i_] == 5], 'Thẻ noble:', [i_ for i_ in range(90,100) if env[:100][i_] == 5])
        print('B_stocks:', env[100:106], 'P1:', env[106:118], 'P2:', env[118:130], 'P3:', env[130:142], 'P4:', env[142:154])
        print('Turn:', env[154], 'Phase:', env[160], 'Nl đã lấy:', env[155:160])
        print('Thẻ đang úp:', 'P1:', [i_ for i_ in range(90) if env[i_] == -1],
        'P1:', [i_ for i_ in range(90) if env[i_] == -2],
        'P2:', [i_ for i_ in range(90) if env[i_] == -3],
        'P3:', [i_ for i_ in range(90) if env[i_] == -4])

    if print_mode:
        _print_()

    _cc = 0
    while env[154] <= 400 and _cc <= 10000:
        p_idx = env[154]%4
        p_state = getAgentState(env, lv1, lv2, lv3)
        list_action = getValidActions(p_state)
        act, per_file = list_player[p_idx](p_state, per_file)
        if list_action[act] != 1:
            raise Exception('Action không hợp lệ')
        stepEnv(act, env, lv1, lv2, lv3)
        if print_mode:
            if act == 0:
                print('Action kết thúc lượt:', act)
            elif act in range(1,4):
                print('Action chọn pha:', act)
            elif act in range(4,9):
                print('Action chọn lấy nguyên liệu:', act-4)
            elif act in range(9,99):
                print('Action chọn úp thẻ:', act-9)
            elif act in range(99, 102):
                print('Action chọn úp ẩn cấp:', act-98)
            elif act in range(102, 192):
                print('Action chọn mua thẻ:', act-102)
            else:
                print('Action chọn trả nguyên liệu:', act-192)

            _print_()

        if checkEnded(env) != 0:
            break

        _cc += 1
    
    if _cc >= 10000:
        print('Chỗ này bị lặp vô tận')
        _print_()
        input()

    turn = env[154]
    for i in range(4):
        env[154] = i
        act, per_file = list_player[i](getAgentState(env, lv1, lv2, lv3), per_file)
    
    env[154] = turn
    return checkEnded(env), per_file


def random_player(p_state, temp_file, per_file):
    arr_action = getValidActions(p_state)
    act_idx = np.random.randint(0, len(arr_action))
    return arr_action[act_idx], temp_file, per_file


@njit()
def numba_one_game(p_lst_idx_shuffle, p0, p1, p2, p3, env, lv1, lv2, lv3, per_file):
    initEnv(env, lv1, lv2, lv3)



    while env[154] <= 400:
        p_idx = env[154]%4
        p_state = getAgentState(env, lv1, lv2, lv3)
        list_action = getValidActions(p_state)
        if p_lst_idx_shuffle[p_idx] == 0:
            act, per_file = p0(p_state, per_file)
        elif p_lst_idx_shuffle[p_idx] == 1:
            act, per_file = p1(p_state, per_file)
        elif p_lst_idx_shuffle[p_idx] == 2:
            act, per_file = p2(p_state, per_file)
        else:
            act, per_file = p3(p_state, per_file)

        if list_action[act] != 1:
            raise Exception('Action không hợp lệ')
        stepEnv(act, env, lv1, lv2, lv3)
        if checkEnded(env) != 0:
            break

    
    turn = env[154]
    for p_idx in range(4):
        env[154] = p_idx
        p_state = getAgentState(env, lv1, lv2, lv3)
        p_state[164] = 1
        if p_lst_idx_shuffle[p_idx] == 0:
            act, per_file = p0(p_state, per_file)
        elif p_lst_idx_shuffle[p_idx] == 1:
            act, per_file = p1(p_state, per_file)
        elif p_lst_idx_shuffle[p_idx] == 2:
            act, per_file = p2(p_state, per_file)
        else:
            act, per_file = p3(p_state, per_file)
    
    env[154] = turn
    return checkEnded(env), per_file


@njit()
def numba_main(p0, p1, p2, p3, num_game,per_file):
    num_won = [0,0,0,0,0]
    p_lst_idx = np.array([0,1,2,3])
    env, lv1, lv2, lv3 = generate()
    for _n in range(num_game):
        np.random.shuffle(p_lst_idx)
        winner, per_file = numba_one_game(p_lst_idx, p0, p1, p2, p3, env, lv1, lv2, lv3, per_file )
        if winner != 0: num_won[p_lst_idx[winner-1]] += 1
        else:num_won[4] += 1

    return num_won, per_file



from numba.typed import List
from base.Splendor.AgentEnv import *



@njit()
def get_func(player_state, id, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11, per12, per13):
    if id == 0: return test2_An_270922(player_state, per0)
    elif id == 1: return test2_Dat_130922(player_state, per1)
    elif id == 2: return test2_Hieu_270922(player_state, per2)
    elif id == 3: return test2_Khanh_270922(player_state, per3)
    elif id == 4: return test2_NhatAnh_270922(player_state, per4)
    elif id == 5: return test2_Phong_130922(player_state, per5)
    elif id == 6: return test2_An_200922(player_state, per6)
    elif id == 7: return test2_Phong_130922(player_state, per7)
    elif id == 8: return test2_Dat_130922(player_state, per8)
    elif id == 9: return test2_Khanh_200922(player_state, per9)
    elif id == 10: return test2_NhatAnh_200922(player_state, per10)
    elif id == 11: return test2_Phong_130922(player_state, per11)
    elif id == 12: return test2_Khanh_130922(player_state, per12)
    else: return test2_Dat_130922(player_state, per13)

@njit()
def one_game_numba(env, lv1, lv2, lv3, p0, list_other, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11, per12, per13):
    initEnv(env, lv1, lv2, lv3)

    while env[154] <= 400:
        idx = env[154]%4
        player_state = getAgentState(env, lv1, lv2, lv3)
        list_action = getValidActions(player_state)
        if list_other[idx] == -1:
            action, per_player = p0(player_state,per_player)
        elif list_other[idx] == -2:
            action = random_Env(player_state)
        else:
            action = get_func(player_state, list_other[idx], per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11, per12, per13)
        
        if list_action[action] != 1:
            raise Exception('Action không hợp lệ')
        stepEnv(action, env, lv1, lv2, lv3)
        if checkEnded(env) != 0:
            break
    
    turn = env[154]
    for idx in range(4):
        env[154] = idx
        if list_other[idx] == -1:
            p_state = getAgentState(env, lv1, lv2, lv3)
            p_state[164] = 1
            act, per_player = p0(p_state, per_player)
    env[154] = turn
    winner = False
    if np.where(list_other == -1)[0] ==  (checkEnded(env) - 1): winner = True
    else: winner = False
    return winner,  per_player



@njit()
def n_game_numba(p0, num_game, per_player, level, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11, per12, per13):
    win = 0
    env, lv1, lv2, lv3 = generate()
    if level == 0:
        list_other = np.array([-2, -2, -2, -1])
    elif level == 1:
        list_other = np.array([8, 13, 7, -1])
    elif level == 2:
        list_other = np.array([3, 12, 2, -1])
    elif level == 3:
        list_other = np.array([1, 4, 9, -1])
    elif level == 4:
        list_other = np.array([5, 6, 0, -1])
    else:
        raise Exception('Hiện tại không có level này')
    for _n in range(num_game):
        np.random.shuffle(list_other)
        winner,per_player  = one_game_numba(env, lv1, lv2, lv3, p0, list_other, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11, per12, per13)
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
    return n_game_numba(p0, n_game, per_player, level, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11, per12, per13)


# @njit()
def one_game_numba_2(env, lv1, lv2, lv3, p0, list_other, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11, per12, per13):
    initEnv(env, lv1, lv2, lv3)

    while env[154] <= 400:
        idx = env[154]%4
        player_state = getAgentState(env, lv1, lv2, lv3)
        list_action = getValidActions(player_state)
        if list_other[idx] == -1:
            action, per_player = p0(player_state,per_player)
        elif list_other[idx] == -2:
            action = random_Env(player_state)
        else:
            action = get_func(player_state, list_other[idx], per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11, per12, per13)
        
        if list_action[action] != 1:
            raise Exception('Action không hợp lệ')
        stepEnv(action, env, lv1, lv2, lv3)
        if checkEnded(env) != 0:
            break
    
    turn = env[154]
    for idx in range(4):
        env[154] = idx
        if list_other[idx] == -1:
            p_state = getAgentState(env, lv1, lv2, lv3)
            p_state[164] = 1
            act, per_player = p0(p_state, per_player)
    env[154] = turn
    winner = False
    if np.where(list_other == -1)[0] ==  (checkEnded(env) - 1): winner = True
    else: winner = False
    return winner,  per_player



# @njit()
def n_game_numba_2(p0, num_game, per_player, level, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11, per12, per13):
    win = 0
    env, lv1, lv2, lv3 = generate()
    if level == 0:
        list_other = np.array([-2, -2, -2, -1])
    elif level == 1:
        list_other = np.array([8, 13, 7, -1])
    elif level == 2:
        list_other = np.array([3, 12, 2, -1])
    elif level == 3:
        list_other = np.array([1, 4, 9, -1])
    elif level == 4:
        list_other = np.array([5, 6, 0, -1])
    else:
        raise Exception('Hiện tại không có level này')
    for _n in range(num_game):
        np.random.shuffle(list_other)
        winner,per_player  = one_game_numba_2(env, lv1, lv2, lv3, p0, list_other, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11, per12, per13)
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
    return n_game_numba_2(p0, n_game, per_player, level, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11, per12, per13)


