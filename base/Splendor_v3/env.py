import numpy as np
import random as rd
from numba import njit
from numba.typed import List
from base.other_func import progress_bar
import warnings
warnings.filterwarnings('ignore')
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning,NumbaExperimentalFeatureWarning, NumbaWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)
warnings.simplefilter('ignore', category=NumbaWarning)

normal_cards_infor = np.array([[0, 2, 2, 2, 0, 0, 0], [0, 2, 3, 0, 0, 0, 0], [0, 2, 1, 1, 0, 2, 1], [0, 2, 0, 1, 0, 0, 2], [0, 2, 0, 3, 1, 0, 1], [0, 2, 1, 1, 0, 1, 1], [1, 2, 0, 0, 0, 4, 0], [0, 2, 2, 1, 0, 2, 0], [0, 1, 2, 0, 2, 0, 1], [0, 1, 0, 0, 2, 2, 0], [0, 1, 1, 0, 1, 1, 1], [0, 1, 2, 0, 1, 1, 1], [0, 1, 1, 1, 3, 0, 0], [0, 1, 0, 0, 0, 2, 1], [0, 1, 0, 0, 0, 3, 0], [1, 1, 4, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 4], [0, 0, 0, 0, 0, 0, 3], [0, 0, 0, 1, 1, 1, 2], [0, 0, 0, 0, 1, 2, 2], [0, 0, 1, 0, 0, 3, 1], [0, 0, 2, 0, 0, 0, 2], [0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 2, 1, 0, 0], [0, 4, 0, 2, 2, 1, 0], [0, 4, 1, 1, 2, 1, 0], [0, 4, 0, 1, 0, 1, 3], [1, 4, 0, 0, 4, 0, 0], [0, 4, 0, 2, 0, 2, 0], [0, 4, 2, 0, 0, 1, 0], [0, 4, 1, 1, 1, 1, 0], [0, 4, 0, 3, 0, 0, 0], [0, 3, 1, 0, 2, 0, 0], [0, 3, 1, 1, 1, 0, 1], [1, 3, 0, 4, 0, 0, 0], [0, 3, 1, 2, 0, 0, 2], [0, 3, 0, 0, 3, 0, 0], [0, 3, 0, 0, 2, 0, 2], [0, 3, 3, 0, 1, 1, 0], [0, 3, 1, 2, 1, 0, 1], [1, 2, 0, 3, 0, 2, 2], [2, 2, 0, 2, 0, 1, 4], [1, 2, 3, 0, 2, 0, 3], [2, 2, 0, 5, 3, 0, 0], [2, 2, 0, 0, 5, 0, 0], [3, 2, 0, 0, 6, 0, 0], [3, 1, 0, 6, 0, 0, 0], [2, 1, 1, 0, 0, 4, 2], [2, 1, 0, 5, 0, 0, 0], [2, 1, 0, 3, 0, 0, 5], [1, 1, 0, 2, 3, 3, 0], [1, 1, 3, 2, 2, 0, 0], [3, 0, 6, 0, 0, 0, 0], [2, 0, 0, 0, 0, 5, 3], [2, 0, 0, 0, 0, 5, 0], [2, 0, 0, 4, 2, 0, 1], [1, 0, 2, 3, 0, 3, 0], [1, 0, 2, 0, 0, 3, 2], [3, 4, 0, 0, 0, 0, 6], [2, 4, 5, 0, 0, 3, 0], [2, 4, 5, 0, 0, 0, 0], [1, 4, 3, 3, 0, 0, 2], [1, 4, 2, 0, 3, 2, 0], [2, 4, 4, 0, 1, 2, 0], [1, 3, 0, 2, 2, 0, 3], [1, 3, 0, 0, 3, 2, 3], [2, 3, 2, 1, 4, 0, 0], [2, 3, 3, 0, 5, 0, 0], [2, 3, 0, 0, 0, 0, 5], [3, 3, 0, 0, 0, 6, 0], [4, 2, 0, 7, 0, 0, 0], [4, 2, 0, 6, 3, 0, 3], [5, 2, 0, 7, 3, 0, 0], [3, 2, 3, 3, 0, 3, 5], [3, 1, 3, 0, 3, 5, 3], [4, 1, 0, 0, 0, 0, 7], [5, 1, 0, 3, 0, 0, 7], [4, 1, 0, 3, 0, 3, 6], [3, 0, 0, 5, 3, 3, 3], [4, 0, 0, 0, 7, 0, 0], [5, 0, 3, 0, 7, 0, 0], [4, 0, 3, 3, 6, 0, 0], [5, 4, 0, 0, 0, 7, 3], [3, 4, 5, 3, 3, 3, 0], [4, 4, 0, 0, 0, 7, 0], [4, 4, 3, 0, 0, 6, 3], [3, 3, 3, 3, 5, 0, 3], [5, 3, 7, 0, 0, 3, 0], [4, 3, 6, 0, 3, 3, 0], [4, 3, 7, 0, 0, 0, 0]])
noble_cards_infor = np.array([[0, 4, 4, 0, 0], [3, 0, 3, 3, 0], [3, 3, 3, 0, 0], [3, 0, 0, 3, 3], [0, 3, 0, 3, 3], [4, 0, 4, 0, 0], [4, 0, 0, 4, 0], [0, 3, 3, 0, 3], [0, 4, 0, 0, 4], [0, 0, 0, 4, 4]])

@njit()
def initEnv():
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

@njit()
def getActionSize():
    return 15

@njit()
def getAgentSize():
    return 4

@njit()
def getStateSize():
    return 161

@njit()
def get_list_id_card_on_lv(lv):
    if len(lv) >= 4:return lv[:4]
    else: return lv[:len(lv)]

@njit()
def concatenate_all_lv_card(lv1, lv2, lv3):
    card_lv1 = normal_cards_infor[get_list_id_card_on_lv(lv1)]
    card_lv2 = normal_cards_infor[get_list_id_card_on_lv(lv2)]
    card_lv3 = normal_cards_infor[get_list_id_card_on_lv(lv3)]
    list_open_card = np.append(card_lv1, card_lv2)
    list_open_card = np.append(list_open_card, card_lv3)
    return list_open_card
    
@njit()
def get_id_card_normal_in_lv(lv1, lv2, lv3):
    list_card_normal_on_board = np.append(get_list_id_card_on_lv(lv1), get_list_id_card_on_lv(lv2))
    list_card_normal_on_board = np.append(list_card_normal_on_board, get_list_id_card_on_lv(lv3))
    return list_card_normal_on_board

@njit()
def getAgentState(env_state, lv1, lv2, lv3):
    p_id = env_state[100] % 4  #Lấy người đang chơi
    b_infor = env_state[101:107] # Lấy 6 loại nguyên liệu của bàn chơi
    p_infor = env_state[107 + 12*p_id:119 + 12*p_id]  #Lấy thông tin người đang chơi, 6 nguyên liệu trên bàn, 5 nguyên liệu mặc định, điểm

    list_open_card = concatenate_all_lv_card(lv1, lv2, lv3) #Lấy list thẻ normal đang mở trên bàn
    list_open_noble = noble_cards_infor[np.where(env_state[90:100] == 5)].flatten() #Lấy list thẻ Noble đang mở trên bàn

    state_card_normal = np.zeros(84)
    state_card_noble = np.zeros(25)
    state_card_normal[:len(list_open_card)] = list_open_card
    state_card_noble[:len(list_open_noble)] = list_open_noble

    list_upside_down_card = normal_cards_infor[np.where(env_state[:90] == -(p_id+1))]
    p_upside_down_card = np.full(21, 0)
    if len(list_upside_down_card) > 0:
        array_hide_card = list_upside_down_card.flatten()
        p_upside_down_card[:len(array_hide_card)] = array_hide_card
    
    st_getting = env_state[155:160] #Lấy thông tin 5 nguyên liệu đang lấy trong turn
    other_scores = [env_state[118 + 12 * id_other_player] for id_other_player in range(4) if id_other_player != p_id] #Lấy điểm của người chơi khác

    p_state = np.zeros(161)
    p_state[0:6] = b_infor # Thông tin của bàn chơi
    p_state[6:18] = p_infor #Thông tin của người chơi
    p_state[18:102] = state_card_normal #Lấy thông tin 12 thẻ đang mở ở trên bàn
    p_state[102:127] = state_card_noble #Thông tin của thẻ noble ở trên bàn
    p_state[127:148] = p_upside_down_card #Lấy thông tin 3 thẻ đang úp
    p_state[148:153] = st_getting #Lấy thông tin 5 nguyên liệu đang lấy trong turn
    p_state[153:156] = other_scores #Lấy điểm của người chơi khác
    p_state[156:159] = (env_state[161:164] != 100) #Lấy thông tin của các thẻ ẩn có thẻ úp, nếu có thể úp thì là 1
    p_state[159] = len(np.where(env_state[:90] == 5)[0]) #Số lượng thẻ có thể úp trong bàn

    cls_game = int(checkEnded(env_state))
    if cls_game == 0:
        p_state[160] = 0
    else:
        p_state[160] = 1

    return p_state.astype(np.float64)

@njit()
def getValidActions(player_state_origin:np.int64):
    list_action_return = np.zeros(15)
    p_upside_down_card =  player_state_origin[127:148] #thông tin 3 thẻ đang úp
    list_action_return[:int(player_state_origin[159])] = 1
    for id_card in range(3):
        card = player_state_origin[127:148][7*id_card: 7+7*id_card]
        if np.sum(card) > 0:
            list_action_return[12+id_card] = 1
    return list_action_return

@njit()
def getReward(p_state):
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


@njit()
def checkEnded(env_state):
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

@njit()
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

@njit()
def get_id_card(card_id):
    if card_id < 40: return 'I', card_id +1
    if 40 <= card_id < 70: return 'II', card_id - 39
    if 70 <= card_id < 90: return 'III', card_id - 69


@njit()
def get_card_can_get(env_state, p_id, cur_p, b_stocks, card_id, nl_auto, nl_bt, card_infor, lv1, lv2, lv3):
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

    env_state[107 + 12*p_id:119 + 12*p_id] = cur_p
    env_state[101:107] = b_stocks

    return env_state, lv1, lv2, lv3
@njit()
def get_card(env_state, action_, cur_p, b_stocks, p_id, lv1, lv2, lv3):
    if action_ < 12:
        id_action = int(action_)
        id_card_normal_ = get_id_card_normal_in_lv(lv1, lv2, lv3)
    else:
        #  print('Thẻ này đang úp')
        id_action = int(action_) - 12
        id_card_normal_ = np.where(env_state[:90] == -(p_id+1))[0]

    card_id = id_card_normal_[id_action]
    card_infor = normal_cards_infor[card_id]
    card_price = card_infor[-5:]
    nl_bo_ra = (card_price>cur_p[6:11]) * (card_price-cur_p[6:11])
    nl_bt = np.minimum(nl_bo_ra, cur_p[:5])
    nl_auto = np.sum(nl_bo_ra - nl_bt)

    card_need = cur_p[:5] + cur_p[6:11] - card_price

    #  print('Taget', get_id_card(card_id))
    if -sum(card_need[np.where(card_need < 0)]) <= cur_p[5] or min(card_need) >= 0: #(x*x>0)
        #  print('Lấy thẻ', card_infor, card_need, get_id_card(card_id))

        return get_card_can_get(env_state, p_id, cur_p, b_stocks, card_id, nl_auto, nl_bt, card_infor, lv1, lv2, lv3)


@njit()
def return_res_to_board(env_state, lv1, lv2, lv3, cur_p, b_stocks, array_res_buy, p_id):
    cur_p[:5] += array_res_buy
    b_stocks[:5] -= array_res_buy
    env_state[107 + 12*p_id:119 + 12*p_id] = cur_p
    env_state[101:107] = b_stocks
    return env_state, lv1, lv2, lv3

@njit()
def return_2_res_to_board(env_state, lv1, lv2, lv3, cur_p, b_stocks, res_, p_id):
    cur_p[:5][res_] += 2
    b_stocks[:5][res_] -= 2

    env_state[107 + 12*p_id:119 + 12*p_id] = cur_p
    env_state[101:107] = b_stocks
    return env_state, lv1, lv2, lv3


@njit()
def stepEnv(action,env_state, lv1, lv2, lv3, all_actions):
    all_actions = np.where(all_actions == 1)[0]
    p_id = env_state[100] % 4
    cur_p = env_state[107 + 12*p_id:119 + 12*p_id]
    b_stocks = env_state[101:107]
    env_state[100] += 1
    if action < 12:
        id_action = action
        id_card_normal = get_id_card_normal_in_lv(lv1, lv2, lv3)
    else:
        #  print('Thẻ này đang úp')
        id_action = action - 12
        id_card_normal = np.where(env_state[:90] == -(p_id+1))[0]

    card_id = id_card_normal[id_action]
    card_infor = normal_cards_infor[card_id]
    card_price = card_infor[-5:]
    nl_bo_ra = (card_price>cur_p[6:11]) * (card_price-cur_p[6:11])
    nl_bt = np.minimum(nl_bo_ra, cur_p[:5])
    nl_auto = np.sum(nl_bo_ra - nl_bt)
    #  print(nl_bo_ra)

    card_need = cur_p[:5] + cur_p[6:11] - card_price

    #  print('Taget', get_id_card(card_id))
    if -np.sum(card_need[np.where(card_need < 0)]) <= cur_p[5] or np.min(card_need) >= 0: #(x*x>0)
        #  print('Lấy thẻ', card_infor, card_need, cur_p[5], get_id_card(card_id))
        return get_card_can_get(env_state, p_id, cur_p, b_stocks, card_id, nl_auto, nl_bt, card_infor, lv1, lv2, lv3)


    res_max = np.argmax(nl_bo_ra)
    if np.sum(cur_p[:6]) <=8:
        if np.max(nl_bo_ra - cur_p[:5]) >= 2 and b_stocks[res_max] >= 4:
            #  print('Lấy 2 nguyên liệu', res_max, card_infor, nl_bo_ra)
            return return_2_res_to_board(env_state, lv1, lv2, lv3, cur_p, b_stocks, res_max, p_id)

        for res in (np.argsort(nl_bo_ra)[::-1]):
            if (nl_bo_ra[res] - cur_p[:5][res]) >= 2 and b_stocks[res] >= 4:
                #  print('Lấy 2 nguyên liệu nhiều thứ 2__', res, card_infor, nl_bo_ra)
                return return_2_res_to_board(env_state, lv1, lv2, lv3, cur_p, b_stocks, res, p_id)


    res_board_have = b_stocks[:5]
    res_can_buy = np.where(((nl_bo_ra - cur_p[:5]) > 0) & (res_board_have > 0), 1, 0)
    array_res_buy = np.full(5, 0)
    if np.sum(cur_p[:6]) <= 7:
        for id, res in enumerate(res_can_buy):
            if res == 1:
                array_res_buy[id] = 1
                if np.sum(array_res_buy) == 3:
                    #  print('Mua nguyên liệu_I:', array_res_buy, '----', res_can_buy, card_infor, nl_bo_ra)
                    return return_res_to_board(env_state, lv1, lv2, lv3, cur_p, b_stocks, array_res_buy, p_id)
                
        res_du = np.where(((res_board_have > 0) - array_res_buy) > 0)[0]
        if np.sum(array_res_buy) == 2 and len(res_du) >= 1:
            res = np.random.randint(0, len(res_du))
            array_res_buy[res_du[res]] = 1
            #  print('Mua nguyên liệu_II:', array_res_buy, '----', res_du, card_infor, nl_bo_ra)
            return return_res_to_board(env_state, lv1, lv2, lv3, cur_p, b_stocks, array_res_buy, p_id)

        if np.sum(array_res_buy) == 1 and len(res_du) >= 2:
            res = np.random.choice(res_du, 2, replace = False)
            array_res_buy[res] = 1
            #  print('Mua nguyên liệu_III:', array_res_buy, '----', res_du, res, card_infor, nl_bo_ra)
            return return_res_to_board(env_state, lv1, lv2, lv3, cur_p, b_stocks, array_res_buy, p_id)


    if len(np.where(env_state[:90] == -(p_id+1))[0]) < 3 and (action < 12): #Úp thẻ
        card_can_upside_down = np.where(env_state[:90] == 5)[0]
        if len(card_can_upside_down) > 0:
            #  print('Úp thẻ', card_infor, card_infor, nl_bo_ra, p_id, get_id_card(card_id))
            env_state[card_id] = -(p_id+1)
            if b_stocks[5] > 0: 
                if np.sum(cur_p[:6]) == 10:
                    for res in np.argsort(nl_bo_ra):
                        if cur_p[res] > 0:
                            #  print('Trả nguyên liệu', res)
                            cur_p[res] -= 1
                            b_stocks[res] += 1
                            break
                if np.sum(cur_p[:6]) <= 9:
                    cur_p[5] += 1
                    b_stocks[5] -= 1
                
            env_state[card_id] = -(p_id+1)
            if card_id < 40:
                env_state, lv1 = get_remove_card_on_lv_and_add_new_card(env_state, lv1,p_id, 161, 3,card_id)
            elif card_id >= 40 and card_id < 70:
                env_state, lv2 = get_remove_card_on_lv_and_add_new_card(env_state, lv2,p_id, 162, 3,card_id)
            else:
                env_state, lv3 = get_remove_card_on_lv_and_add_new_card(env_state, lv3,p_id, 163, 3,card_id)

            env_state[107 + 12*p_id:119 + 12*p_id] = cur_p
            env_state[101:107] = b_stocks
            return env_state, lv1, lv2, lv3



    action_have_res_count = np.array([-99])
    soluong_nl_bo_ra = np.array([99])
    action_co_the_lay_the = np.array([-1])

    for action_ in all_actions:  #Chọn những thẻ có thể lấy trên bàn và đang úp
        if action_ < 12:
            id_action_ = int(action_)
            id_card_normal_ = get_id_card_normal_in_lv(lv1, lv2, lv3)
        else:
            id_action_ = int(action_) - 12
            id_card_normal_ = np.where(env_state[:90] == -(p_id+1))[0]

        card_id_ = id_card_normal_[id_action_]
        card_infor_ = normal_cards_infor[card_id_]
        card_price_ = card_infor_[-5:]
        nl_bo_ra_ = (card_price_>cur_p[6:11]) * (card_price_-cur_p[6:11])
        card_need_ = cur_p[:5] + cur_p[6:11] - card_price_

        if -np.sum(card_need_[np.where(card_need_ < 0)]) <= cur_p[5] or np.min(card_need_) >= 0: #(x*x>0)
            action_co_the_lay_the = np.append(action_co_the_lay_the, action_)
            soluong_nl_bo_ra = np.append(soluong_nl_bo_ra, np.sum(nl_bo_ra_))
            #  print('Thẻ có thể lấy', get_id_card(card_id_), 'Số lượng nguyên liệu bỏ ra', np.sum(nl_bo_ra_))
            if (nl_bo_ra[card_infor_[1]] - cur_p[card_infor_[1]])> 0:
                #  print('Thẻ này có nguyên liệu cần')
                action_have_res_count = np.append(action_have_res_count, action_)
            else:
                action_have_res_count = np.append(action_have_res_count, -99)
    
    action_chon_lay_the_co_nl_mac_dinh = np.where(action_have_res_count != -99)[0]
    check = False
    if len(action_chon_lay_the_co_nl_mac_dinh) > 0: #Lấy thẻ có nguyên liệu mặc định
        nl_bo_ra_min = 99
        #  print("Lấy thẻ có nguyên liệu mặc định", soluong_nl_bo_ra, action_chon_lay_the_co_nl_mac_dinh)
        for action_id in action_chon_lay_the_co_nl_mac_dinh:
            if soluong_nl_bo_ra[action_id] < nl_bo_ra_min:
                nl_bo_ra_min = soluong_nl_bo_ra[action_id]
                action_ = action_have_res_count[action_id]
                check = True
        if check == True:
            return get_card(env_state, action_, cur_p, b_stocks, p_id, lv1, lv2, lv3)


    sum_res_need = 10 - np.sum(cur_p[:6])
    if sum_res_need >= 3: sum_res_need = 3

    res_board_have = b_stocks[:5]
    res_can_buy = np.where(((nl_bo_ra - cur_p[:5]) > 0) & (res_board_have > 0), 1, 0)
    array_res_buy = np.full(5, 0)

    for id, res in enumerate(res_can_buy):
        if res == 1:
            if np.sum(array_res_buy) == sum_res_need:
                break
            array_res_buy[id] = 1

    if np.sum(array_res_buy) > 1 :
        #  print('Mua nguyên liệu_IIII:', array_res_buy, '----', res_can_buy, card_infor, nl_bo_ra)
        return return_res_to_board(env_state, lv1, lv2, lv3, cur_p, b_stocks, array_res_buy, p_id)

    if len(soluong_nl_bo_ra) > 1 and np.min(soluong_nl_bo_ra) == 0:
        action_ = action_co_the_lay_the[np.argmin(soluong_nl_bo_ra)]
        #  print("Lấy thẻ không có nguyên liệu mặc định miễn phí:", np.min(soluong_nl_bo_ra))
        return get_card(env_state, action_, cur_p, b_stocks, p_id, lv1, lv2, lv3)

    if np.max(nl_bo_ra) > 0 and b_stocks[res_max] >= 4 and np.sum(cur_p[:6]) <= 8:
        #  print('Lấy 2 nguyên liệu_II:', res_max, '----', card_infor, nl_bo_ra)
        return return_2_res_to_board(env_state, lv1, lv2, lv3, cur_p, b_stocks, res_max, p_id)

    if np.sum(array_res_buy) > 0:
        for id, res in enumerate(b_stocks[:5]):
            if res > 0:
                if np.sum(array_res_buy) == sum_res_need:
                    break
                array_res_buy[id] = 1
        #  print('Mua nguyên liệu_IIIII:', array_res_buy, '----', res_can_buy, card_infor, nl_bo_ra)
        return return_res_to_board(env_state, lv1, lv2, lv3, cur_p, b_stocks, array_res_buy, p_id)

    res_max = np.argmax(b_stocks[:5])
    if b_stocks[res_max] >= 4 and np.sum(cur_p[:6]) <= 8:
        #  print('Lấy 2 nguyên liệu_III, bất kỳ:', res_max, '----', card_infor, nl_bo_ra)
        return return_2_res_to_board(env_state, lv1, lv2, lv3, cur_p, b_stocks, res_max, p_id)

    for id, res in enumerate(b_stocks[:5]):
        if res > 0:
            if np.sum(array_res_buy) == sum_res_need:
                break
            array_res_buy[id] = 1

    if np.sum(array_res_buy) > 0:
        #  print('Mua nguyên liệu còn thừa trên bàn_IIIIII:', array_res_buy, '----', res_can_buy, card_infor, nl_bo_ra, sum_res_need)
        return return_res_to_board(env_state, lv1, lv2, lv3, cur_p, b_stocks, array_res_buy, p_id)

    if len(soluong_nl_bo_ra) > 1:
        action_ = action_co_the_lay_the[np.argmin(soluong_nl_bo_ra)]
        #  print("Lấy thẻ không có nguyên liệu mặc định:", np.min(soluong_nl_bo_ra))
        return get_card(env_state, action_, cur_p, b_stocks, p_id, lv1, lv2, lv3)

    
    #  print('Không làm gì cả')
    return env_state, lv1, lv2, lv3



list_color = ['red', 'blue', 'green', 'black', 'white', 'auto_color']

def _print_(env, lv1, lv2, lv3):
    #  print('Lượt của người chơi:', env[100]%4 - 1, list_color)
    #  print('B_stocks:', env[101:107], 'Turn:', env[100], )
    #  print('Thẻ 1:', [i_+1 for i_ in get_list_id_card_on_lv(lv1)], list(lv1+1))
    #  print('Thẻ 2:', [i_-39 for i_ in get_list_id_card_on_lv(lv2)], list(lv2-39))
    #  print('Thẻ 3:', [i_-69 for i_ in get_list_id_card_on_lv(lv3)], list(lv3-69))
    #  print('Noble:', [i_-89 for i_ in range(90,100) if env[:100][i_] == 5])
    #  print('P1:', env[107:113], env[113:118], env[118], [get_id_card(i_) for i_ in range(90) if env[i_] == -1], [get_id_card(i_) for i_ in range(90) if env[i_] == 1], '\nP2:', env[119:125], env[125:130], env[130], [get_id_card(i_) for i_ in range(90) if env[i_] == -2], [get_id_card(i_) for i_ in range(90) if env[i_] == 2],'\nP3:', env[131:137], env[137:142], env[142], [get_id_card(i_) for i_ in range(90) if env[i_] == -3], [get_id_card(i_) for i_ in range(90) if env[i_] == 3],'\nP4:', env[143:149], env[149:154], env[154], [get_id_card(i_) for i_ in range(90) if env[i_] == -4], [get_id_card(i_) for i_ in range(90) if env[i_] == 4],)
    pass

def one_game(list_player, per_file):
    env, lv1, lv2, lv3 = initEnv()
    temp_file = [[0],[0],[0],[0]]
    _cc = 0
    while env[100] <= 400 and _cc <= 10000:
        p_idx = env[100]%4
        p_state = getAgentState(env, lv1, lv2, lv3)
        #  print('-----------------------------------------------------------------------')
        act, temp_file[p_idx], per_file = list_player[p_idx](p_state, temp_file[p_idx], per_file)
        list_action = getValidActions(p_state)
        if list_action[act] != 1:
            raise Exception('Action không hợp lệ')
        env, lv1, lv2, lv3 = stepEnv(act, env, lv1, lv2, lv3, list_action)
        #  print('-----')
        _print_(env, lv1, lv2, lv3)
        if checkEnded(env) != 0:
            break

        _cc += 1
    
    turn = env[100]
    for i in range(4):
        env[100] = i
        p_state = getAgentState(env, lv1, lv2, lv3)
        p_state[160] = 1
        act, temp_file[i], per_file = list_player[i](p_state, temp_file[i], per_file)
    
    env[100] = turn
    return checkEnded(env), per_file

def normal_main(list_player, num_game,per_file):
    if len(list_player) != 4:
        #  print('Game chỉ cho phép có đúng 4 người chơi')
        return [-1,-1,-1,-1,-1], per_file
    
    num_won = [0,0,0,0,0]
    p_lst_idx = [0,1,2,3]
    for _n in range(num_game):
        progress_bar(_n, num_game)
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


@njit()
def numba_one_game(p_lst_idx_shuffle, p0, p1, p2, p3, per_file):
    env, lv1, lv2, lv3 = initEnv()
    _cc = 0

    temp_1_player = List()
    temp_1_player.append(np.array([[0.]]))
    temp_file = [temp_1_player]*(getAgentSize())

    while env[100] <= 400 and _cc <= 10000:
        p_idx = env[100]%4
        p_state = getAgentState(env, lv1, lv2, lv3)
        if p_lst_idx_shuffle[p_idx] == 0:
            act, temp_file[p_idx], per_file = p0(p_state, temp_file[p_idx], per_file)
        elif p_lst_idx_shuffle[p_idx] == 1:
            act, temp_file[p_idx], per_file = p1(p_state, temp_file[p_idx], per_file)
        elif p_lst_idx_shuffle[p_idx] == 2:
            act, temp_file[p_idx], per_file = p2(p_state, temp_file[p_idx], per_file)
        else:
            act, temp_file[p_idx], per_file = p3(p_state, temp_file[p_idx], per_file)

        list_action = getValidActions(p_state)
        if list_action[act] != 1:
            raise Exception('Action không hợp lệ')
            
        env, lv1, lv2, lv3 = stepEnv(act, env, lv1, lv2, lv3, list_action)

        if checkEnded(env) != 0:
            break

        _cc += 1
    
    turn = env[100]
    for p_idx in range(4):
        env[100] = p_idx
        p_state = getAgentState(env, lv1, lv2, lv3)
        p_state[160] = 1
        if p_lst_idx_shuffle[p_idx] == 0:
            act, temp_file[p_idx], per_file = p0(p_state, temp_file[p_idx], per_file)
        elif p_lst_idx_shuffle[p_idx] == 1:
            act, temp_file[p_idx], per_file = p1(p_state, temp_file[p_idx], per_file)
        elif p_lst_idx_shuffle[p_idx] == 2:
            act, temp_file[p_idx], per_file = p2(p_state, temp_file[p_idx], per_file)
        else:
            act, temp_file[p_idx], per_file = p3(p_state, temp_file[p_idx], per_file)
    
    env[100] = turn
    return checkEnded(env), per_file


@njit()
def numba_main(p0, p1, p2, p3, num_game,per_file):
    num_won = [0,0,0,0,0]
    p_lst_idx = np.array([0,1,2,3])
    for _n in range(num_game):
        # progress_bar(_n, num_game)
        np.random.shuffle(p_lst_idx)
        winner, per_file = numba_one_game(p_lst_idx, p0, p1, p2, p3, per_file )
        if winner != 0: num_won[p_lst_idx[winner-1]] += 1
        else:num_won[4] += 1
    return num_won, per_file


# @njit()
# def get_func(player_state, id, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11):
#     if id == 0: return test2_An_270922(player_state, per0)
#     elif id == 1: return test2_Dat_130922(player_state, per1)
#     elif id == 2: return test2_Hieu_270922(player_state, per2)
#     elif id == 3: return test2_Khanh_270922(player_state, per3)
#     elif id == 4: return test2_Phong_130922(player_state, per4)
#     elif id == 5: return test2_An_200922(player_state, per5)
#     elif id == 6: return test2_Phong_130922(player_state, per6)
#     elif id == 7: return test2_Dat_130922(player_state, per7)
#     elif id == 8: return test2_Khanh_200922(player_state, per8)
#     elif id == 9: return test2_Khanh_130922(player_state, per9)
#     elif id == 10: return test2_Dat_130922(player_state, per10)
#     else: return test2_Hieu_130922(player_state, per11)

@njit()
def random_Env(p_state):
    p_state = p_state[:-1]
    arr_action = getValidActions(p_state)
    arr_action = np.where(arr_action == 1)[0]
    act_idx = np.random.randint(0, len(arr_action))
    return arr_action[act_idx]

@njit()
def one_game_numba(p0, list_other, per_player):
    env, lv1, lv2, lv3 = initEnv()

    _temp_ = List()
    _temp_.append(np.array([[0]]))

    _cc = 0
    while env[100] <= 400 and _cc <= 10000:
        idx = env[100]%4
        player_state = getAgentState(env, lv1, lv2, lv3)
        if list_other[idx] == -1:
            action, _temp_, per_player = p0(player_state,_temp_,per_player)
        elif list_other[idx] == -2:
            action = random_Env(player_state)
        # else:
        #     action = get_func(player_state, list_other[idx], per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11)

        list_action = getValidActions(player_state)
        if list_action[action] != 1:
            raise Exception('Action không hợp lệ')

        env, lv1, lv2, lv3 = stepEnv(action, env, lv1, lv2, lv3, list_action)
        if checkEnded(env) != 0:
            break
        
        _cc += 1
    turn = env[100]
    for idx in range(4):
        env[100] = idx
        if list_other[idx] == -1:
            p_state = getAgentState(env, lv1, lv2, lv3)
            p_state[160] = 1
            act, _temp_, per_player = p0(p_state, _temp_, per_player)
    env[100] = turn
    winner = False
    if np.where(list_other == -1)[0] ==  (checkEnded(env) - 1): winner = True
    else: winner = False
    return winner,  per_player


@njit()
def n_game_numba(p0, num_game, per_player, level):
    win = 0
    if level == 0:
        list_other = np.array([-2, -2, -2, -1])
    else:
        raise Exception('Hiện tại không có level này')
    for _n in range(num_game):
        np.random.shuffle(list_other)
        winner,per_player  = one_game_numba(p0, list_other, per_player)
        win += winner
    return win, per_player



def numba_main_2(p0, n_game, per_player, level):
    # list_all_players = dict_game_for_player[game_name_]
    # list_data = load_data_per2(list_all_players, game_name_)
    # per0 = list_data[0]
    # per1 = list_data[1]
    # per2 = list_data[2]
    # per3 = list_data[3]
    # per4 = list_data[4]
    # per5 = list_data[5]
    # per6 = list_data[6]
    # per7 = list_data[7]
    # per8 = list_data[8]
    # per9 = list_data[9]
    # per10 = list_data[10]
    # per11 = list_data[11]
    return n_game_numba(p0, n_game, per_player, level)



@njit()
def random_Env(p_state):
    p_state = p_state[:-1]
    arr_action = getValidActions(p_state)
    arr_action = np.where(arr_action == 1)[0]
    act_idx = np.random.randint(0, len(arr_action))
    return arr_action[act_idx]

# @njit()
def one_game_numba_2(p0, list_other, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11):
    env, lv1, lv2, lv3 = initEnv()

    _temp_ = List()
    _temp_.append(np.array([[0]]))

    _cc = 0
    while env[100] <= 400 and _cc <= 10000:
        idx = env[100]%4
        player_state = getAgentState(env, lv1, lv2, lv3)
        if list_other[idx] == -1:
            action, _temp_, per_player = p0(player_state,_temp_,per_player)
        elif list_other[idx] == -2:
            action = random_Env(player_state)
        # else:
        #     action = get_func(player_state, list_other[idx], per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10, per11)

        list_action = getValidActions(player_state)
        if list_action[action] != 1:
            raise Exception('Action không hợp lệ')

        env, lv1, lv2, lv3 = stepEnv(action, env, lv1, lv2, lv3, list_action)
        if checkEnded(env) != 0:
            break
        
        _cc += 1
    turn = env[100]
    for idx in range(4):
        env[100] = idx
        if list_other[idx] == -1:
            p_state = getAgentState(env, lv1, lv2, lv3)
            p_state[160] = 1
            act, _temp_, per_player = p0(p_state, _temp_, per_player)
    env[100] = turn
    winner = False
    if np.where(list_other == -1)[0] ==  (checkEnded(env) - 1): winner = True
    else: winner = False
    return winner,  per_player


# @njit()
def n_game_numba_2(p0, num_game, per_player, level):
    win = 0
    if level == 0:
        list_other = np.array([-2, -2, -2, -1])
    else:
        raise Exception('Hiện tại không có level này')
    for _n in range(num_game):
        np.random.shuffle(list_other)
        winner,per_player  = one_game_numba_2(p0, list_other, per_player)
        win += winner
    return win, per_player



def normal_main_2(p0, n_game, per_player, level):
    # list_all_players = dict_game_for_player[game_name_]
    # list_data = load_data_per2(list_all_players, game_name_)
    # per0 = list_data[0]
    # per1 = list_data[1]
    # per2 = list_data[2]
    # per3 = list_data[3]
    # per4 = list_data[4]
    # per5 = list_data[5]
    # per6 = list_data[6]
    # per7 = list_data[7]
    # per8 = list_data[8]
    # per9 = list_data[9]
    # per10 = list_data[10]
    # per11 = list_data[11]
    return n_game_numba_2(p0, n_game, per_player, level)
