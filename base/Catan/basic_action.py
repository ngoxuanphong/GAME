from base.Catan.index import*
from numba import jit, njit
import random
import numpy as np



@njit()
def RoadCanBuild(road_have, road_not_yet_have, id_action, house_data):

    lst_point_other_player = list_point_other_player(id_action, house_data)
    list_road_can_build = [-1]
    point_1_road = [-1]
    for road in road_have:
        for road_near in ROAD_ROAD_RELATIVE[road]:
            if road_near in road_not_yet_have:
                for point in ROAD_BY_POINT[road_near]:
                    if point in ROAD_BY_POINT[road] and point not in lst_point_other_player:
                        list_road_can_build.append(road_near)
                        point_1_road.append(point)
                        break

    list_road_can_build = np.delete(list_road_can_build, 0)
    point_1_road = np.delete(point_1_road, 0)
    return np.unique(list_road_can_build), np.unique(point_1_road)

@njit()
def Point2Road(player_state, road_build):
    list_action_build_road = [-1]
    for i in road_build:
        if player_state[P_TEMP_POINT] in ROAD_BY_POINT[i]:
            id_point = np.where(ROAD_BY_POINT[i] == player_state[P_TEMP_POINT])[0][0]
            if id_point == 0:
                list_action_build_road.append(ROAD_BY_POINT[i][1])
            else:
                list_action_build_road.append(ROAD_BY_POINT[i][0])
    if len(list_action_build_road) > 1:
        list_action_build_road.pop(0)
    return np.array(list_action_build_road)

@njit()
def HouseCanBuild(house_list, road_list, turn):
    house_built_mine = len(np.where(house_list==0)[0])
    if house_built_mine == 5:
        return np.array([-1])
    house_built = np.where(house_list!=-1)[0]
    point = np.ones(len(house_list))
    position_point_cant_used_house = house_built
    # position_point_cant_used_house = list(house_built)
    for index_house in house_built:
        # position_point_cant_used_house = position_point_cant_used_house + list(POINT_POINT_RELATIVE[index_house])
        position_point_cant_used_house = np.append(position_point_cant_used_house,POINT_POINT_RELATIVE[index_house])
    position_point_cant_used_house = np.unique(position_point_cant_used_house)
    position_point_cant_used_house = np.delete(position_point_cant_used_house, np.where(position_point_cant_used_house==-1)[0])
    point[position_point_cant_used_house] = np.zeros(len(position_point_cant_used_house))
    position_point_can_used_road = np.where(point == 1)[0]
    if turn > 8:
        for index_house in position_point_can_used_road:
            check = False
            for i_house_road in POINT_ROAD_RELATIVE[index_house]:
                if i_house_road != -1 and road_list[i_house_road] == 0:
                    check = True
            if check == True:
                pass
            else:
                point[index_house] = 0
    return_data = np.where(point==1)[0]
    return return_data

@njit()
def dung_the_knight(env_state):
        #dùng knight
        id_action = env_state[ID_ACTION]
        env_state[int(ALL_INFOR_PLAYER*id_action + INDEX_NUMBER_KNIGHT_USED_IN_ATTRIBUTE)] += 1
        #đủ 3 knight mới xét
        if env_state[int(ALL_INFOR_PLAYER*id_action + INDEX_NUMBER_KNIGHT_USED_IN_ATTRIBUTE)] > 2:
            if env_state[LARGEST_ARMY_PLAYER] == -1:
                env_state[LARGEST_ARMY_PLAYER] = id_action
                env_state[int(ALL_INFOR_PLAYER*id_action)] += 2
            else:
                max_knight_use = env_state[int(ALL_INFOR_PLAYER*env_state[LARGEST_ARMY_PLAYER]+INDEX_NUMBER_KNIGHT_USED_IN_ATTRIBUTE)]
                if max_knight_use < env_state[int(ALL_INFOR_PLAYER*id_action + INDEX_NUMBER_KNIGHT_USED_IN_ATTRIBUTE)] and id_action != env_state[LARGEST_ARMY_PLAYER]:
                    #nếu người dùng knight vượt qua người nhiều knight nhất hiện tại
                    env_state[int(ALL_INFOR_PLAYER*env_state[LARGEST_ARMY_PLAYER])] -= 2
                    env_state[int(ALL_INFOR_PLAYER*id_action)] += 2
                    env_state[LARGEST_ARMY_PLAYER] = id_action
                else:
                    pass
        else:
            pass
        return env_state

@njit() 
def GetMaterialFirst(deck_list,point):
    list_ = np.array([0,0,0,0,0])
    for block in range(len(POINT_IN_BLOCK)):
        if point in POINT_IN_BLOCK[block]:
            if deck_list[block] != 0:
                list_[int(deck_list[block]-1)] += 1
    return list_

@njit()
def CheckMaterialForAction(deck_list,type_):
    if type_==1:
        for i in deck_list[:4]:
            if i <= 0:
                return False
        return True
    elif type_ == 2:
        if deck_list[4] >=3 and deck_list[3] >= 2:
            return True
        return False
    elif type_ == 3:
        if deck_list[1] >=1 and deck_list[0] >= 1:
            return True
        return False
    elif type_ == 4:
        for i in deck_list[2:]:
            if i <= 0:
                return False
        return True

@njit()
def updateSumMaterial(env_state):
    env_state[ATTRIBUTE_PLAYER_1_INDEX + 1] = np.sum(env_state[CARD_1_PLAYER_INDEX : CARD_1_PLAYER_INDEX + CARD_BANK_LEN])
    env_state[ATTRIBUTE_PLAYER_2_INDEX + 1] = np.sum(env_state[CARD_2_PLAYER_INDEX : CARD_2_PLAYER_INDEX + CARD_BANK_LEN])
    env_state[ATTRIBUTE_PLAYER_3_INDEX + 1] = np.sum(env_state[CARD_3_PLAYER_INDEX : CARD_3_PLAYER_INDEX + CARD_BANK_LEN])
    env_state[ATTRIBUTE_PLAYER_4_INDEX + 1] = np.sum(env_state[CARD_4_PLAYER_INDEX : CARD_4_PLAYER_INDEX + CARD_BANK_LEN])
    return env_state

@njit()
def remote_a_if_a_in_b(a, b):
    list_index = np.array([i for i in range(len(a)) if a[i] in b]).astype(np.int64)
    return np.delete(a,list_index)

@njit()
def list_point_other_player(id_action, house_data):
    list_point_of_other_player = []
    for i in range(len(house_data)):
        if house_data[i] != id_action and house_data[i] != -1 and house_data[i] != id_action + 4:
            list_point_of_other_player.append(int(i))
    return list_point_of_other_player

@njit()
def process_return_roll(env_state):
    number_to_roll = env_state[TOKEN_BLOCK_INDEX:TYPE_SOURCE_INDEX]
    all_type_source = env_state[TYPE_SOURCE_INDEX:TYPE_HARBOR_INDEX]
    card_bank = env_state[CARD_BANK_INDEX:CARD_EFFECT_BANK_INDEX]
    all_player_map = env_state[POINT_INDEX:ROAD_INDEX]
    block_robber = env_state[ROBBER_BLOCK_INDEX]
    roll_dice = env_state[LAST_ROLL]
    block_source = np.where(number_to_roll == roll_dice)[0]
    block_source = block_source[block_source!= block_robber]            #loại bỏ ô chưa robber
    type_source_return = all_type_source[block_source]
    if len(block_source) > 1:
        point_in_block1 = all_player_map[POINT_IN_BLOCK[block_source[0]]]
        point_in_block2 = all_player_map[POINT_IN_BLOCK[block_source[1]]]
        if type_source_return[0] == type_source_return[1]:
            point_in_2_block = np.append(point_in_block1, point_in_block2)
            source_for_player = return_source(point_in_2_block)
            source_type = int(type_source_return[0] - 1)
            if np.sum(source_for_player) <= card_bank[source_type]:
                #nếu ngân hàng đủ để trả
                env_state[CARD_1_PLAYER_INDEX+source_type] += source_for_player[0]
                env_state[CARD_2_PLAYER_INDEX+source_type] += source_for_player[1]
                env_state[CARD_3_PLAYER_INDEX+source_type] += source_for_player[2]
                env_state[CARD_4_PLAYER_INDEX+source_type] += source_for_player[3]
                env_state[CARD_BANK_INDEX + source_type] -= np.sum(source_for_player)
        else:
            #trả cho block1
            source_for_player1 = return_source(point_in_block1)
            source_type = int(type_source_return[0] - 1)
            if np.sum(source_for_player1) <= card_bank[source_type]:
                #nếu ngân hàng đủ để trả
                env_state[CARD_1_PLAYER_INDEX+source_type] += source_for_player1[0]
                env_state[CARD_2_PLAYER_INDEX+source_type] += source_for_player1[1]
                env_state[CARD_3_PLAYER_INDEX+source_type] += source_for_player1[2]
                env_state[CARD_4_PLAYER_INDEX+source_type] += source_for_player1[3]
                env_state[CARD_BANK_INDEX + source_type] -= np.sum(source_for_player1)
            #trả cho block 2
            source_for_player2 = return_source(point_in_block2)
            source_type = int(type_source_return[1] - 1)
            if np.sum(source_for_player2) <= card_bank[source_type]:
                #nếu ngân hàng đủ để trả
                env_state[CARD_1_PLAYER_INDEX+source_type] += source_for_player2[0]
                env_state[CARD_2_PLAYER_INDEX+source_type] += source_for_player2[1]
                env_state[CARD_3_PLAYER_INDEX+source_type] += source_for_player2[2]
                env_state[CARD_4_PLAYER_INDEX+source_type] += source_for_player2[3]
                env_state[CARD_BANK_INDEX + source_type] -= np.sum(source_for_player2)
    elif len(block_source) == 1:
        point_in_block1 = all_player_map[POINT_IN_BLOCK[block_source[0]]]
        source_for_player = return_source(point_in_block1)
        source_type = int(type_source_return[0] - 1)
        if np.sum(source_for_player) <= card_bank[source_type]:
            #nếu ngân hàng đủ để trả
            env_state[CARD_1_PLAYER_INDEX+source_type] += source_for_player[0]
            env_state[CARD_2_PLAYER_INDEX+source_type] += source_for_player[1]
            env_state[CARD_3_PLAYER_INDEX+source_type] += source_for_player[2]
            env_state[CARD_4_PLAYER_INDEX+source_type] += source_for_player[3]
            env_state[CARD_BANK_INDEX + source_type] -= np.sum(source_for_player)
    else:
        pass
    #cập nhật tổng số lượng thẻ nguyên liệu của người chơi
    env_state = updateSumMaterial(env_state)
        
    return env_state

@njit()
def return_source(point_in_block):
    source_for_player = np.zeros(4)
    point_in_block_need = point_in_block[point_in_block>-1]
    if len(point_in_block_need) > 0:   
        player_return = point_in_block_need%4
        number_source_return = np.add((point_in_block_need > -1).astype(np.int64), (point_in_block_need > 3).astype(np.int64))  
        # print('check', number_source_return, point_in_block)
        for i in range(len(player_return)):
            source_for_player[int(player_return[i])] += number_source_return[i]
        # print('taif nguyen', source_for_player)
        return source_for_player.astype(np.int64)
    else:
        return source_for_player.astype(np.int64)

@njit()
def check_trade_with_harbor(env_state):
    main_deal = env_state[OFFER_MAIN_INDEX : OFFER_MAIN_INDEX + OFFER_LEN]
    all_player_map = env_state[POINT_INDEX : POINT_INDEX + POINT_LEN]
    all_harbor = env_state[TYPE_HARBOR_INDEX : TYPE_HARBOR_INDEX + TYPE_HARBOR_LEN]
    all_settle_and_city_index = np.where((all_player_map > -1) & (all_player_map % 4 == env_state[ID_ACTION]))[0]
    all_settle_and_city_at_harbor = np.array([-1])
    for harbor in all_settle_and_city_index:
        if harbor in POINT_IN_BLOCK:
            all_settle_and_city_at_harbor = np.append(all_settle_and_city_at_harbor, harbor)
    type_trade = np.max(main_deal)
    type_source = np.argmax(main_deal)
    number_return = np.max(main_deal[CARD_BANK_LEN:])       #số lượng tài nguyên muốn lấy
    if len(all_settle_and_city_at_harbor) > 1:
        for harbor in all_settle_and_city_at_harbor:
            if harbor in POINT_IN_HARBOR:
                type_harbor = all_harbor[np.where(POINT_IN_HARBOR == harbor)[0]//2]
                if type_harbor == 0:
                    if type_trade%3 == 0 and type_trade/number_return == 3:
                        return True
                else:
                    if type_trade%2 == 0 and type_source == (type_harbor - 1) and number_return == 1:
                        return True
            else:
                continue
    if type_trade %4 == 0 and type_trade/4 == number_return:
        return True
    else:
        return False

def print_mode_action(action, env_state):
    print('--------------------------------------------------------')
    print('TURN:', int(env_state[TURN]), '|  Người chơi:', int(env_state[ID_ACTION]), end = ' ')
    print('|  PHASE :', int(env_state[PHASE]), '|  ACTION:', action, ' |   CARD', list(env_state[CARD_BANK_INDEX : CARD_BANK_INDEX + CARD_BANK_LEN].astype(np.int64)))
    if action in range(54, 73): print('Chọn ô muốn đặt robble', action - 54)  
    if action in range(73, 78): print('Chọn nguyên liệu bỏ khi chia bài', action - 73) 
    if action in range(78, 81): print('Chọn người để cướp tài nguyên', action - 78) 
    if action in range(81, 86): print('Lấy nguyên liệu dem đi trading', action - 81)
    if action in range(86, 91): print('Chọn nguyên liệu muốn nhận khi trading', action - 86)
    if action == 95: print('Dùng thẻ xây 2 đường')
    if action == 96: print('Dùng thẻ lấy 2 nguyên liệu')
    if action == 98: print('Chọn action xây nhà')
    if action == 99: print('Chọn action upgrade nhà')
    if action == 100: print('Chọn action xây đường')
    if action == 101: print('Chọn action mua thẻ dev')
    if action == 102: print('Đổ xúc xắc')
    if action == 103: print('Bỏ lượt')
    if action == 104: print('Người chơi không đồng ý trade')
    if action == 105: print('Người chơi đồng ý trade')
    if action == 106: print('Action đi trading')
    if action == 107: print('Action đi trading cảng')
    if env_state[PHASE] == 3: print('Xây nhà số', action)
    if env_state[PHASE] == 5: print('Xây đường điểm số 1:', action)
    if env_state[PHASE] == 4: print('Xây thành phố điểm số:', action)
    if env_state[PHASE] == 6: print('Xây đường điểm số 2:', action)

def print_mode_board(action, env_state):
    print('*********')
    print(list(env_state[OTHER_IN4_INDEX:OTHER_IN4_INDEX+OTHER_IN4_LEN].astype(np.int64)))
    
    for i in range(4):
        print(list(env_state[ALL_INFOR_PLAYER*i + CARD_1_PLAYER_INDEX:ALL_INFOR_PLAYER*i + CARD_1_PLAYER_INDEX+ CARD_BANK_LEN].astype(np.int64)), end = ' | ')
        print(list(env_state[ALL_INFOR_PLAYER*i:ALL_INFOR_PLAYER*i+ATTRIBUTE_PLAYER].astype(np.int64)), end = ' | ')
        print('ROAD: ', list(np.where(env_state[ROAD_INDEX:ROAD_INDEX+ROAD_LEN] == i )[0].astype(np.int64)), end = ' | ')
        print('POINT: ', list(np.where(env_state[POINT_INDEX:POINT_INDEX+POINT_LEN] == i )[0].astype(np.int64)), end = ' | ')
        print('DEV: ', list(env_state[ALL_INFOR_PLAYER*i + CARD_EFFECT_1_PLAYER_INDEX:ALL_INFOR_PLAYER*i + CARD_EFFECT_1_PLAYER_INDEX + CARD_EFFECT_PLAYER_LEN].astype(np.int64)))

def check_all_source(env_state, action, phase):
    all_source = np.zeros(5)
    sum_all_source = 0
    for i in range(4):
        all_source = all_source + env_state[ALL_INFOR_PLAYER*i+CARD_1_PLAYER_INDEX:ALL_INFOR_PLAYER*i+CARD_1_PLAYER_INDEX+CARD_BANK_LEN]
        sum_all_source += env_state[ALL_INFOR_PLAYER*i+1]
    all_source += env_state[CARD_BANK_INDEX:CARD_BANK_INDEX+CARD_BANK_LEN]
    sum_all_source += np.sum(env_state[CARD_BANK_INDEX:CARD_BANK_INDEX+CARD_BANK_LEN])
    if np.min(all_source) != 19 or np.max(all_source) != 19 or sum_all_source != 95:
        print(f'toang ở phase {phase} action {action}')
        raise Exception('toang tông tai nguyen tren ban')

@njit()
def check_winner(env_state):
    score = env_state[ATTRIBUTE_PLAYER_1_INDEX:ATTRIBUTE_PLAYER_4_INDEX+2:ALL_INFOR_PLAYER]
    vitory_card = env_state[CARD_EFFECT_1_PLAYER_INDEX+CARD_EFFECT_PLAYER_LEN-1:CARD_EFFECT_4_PLAYER_INDEX+CARD_EFFECT_PLAYER_LEN+1:ALL_INFOR_PLAYER]
    score_real = score+vitory_card
    max_score = np.max(score_real)
    if max_score < 10:
        return -1
    return np.where(score_real==max_score)[0][-1]



@njit()
def system_check_end(env_state):
    score = env_state[ATTRIBUTE_PLAYER_1_INDEX:ATTRIBUTE_PLAYER_4_INDEX+2:ALL_INFOR_PLAYER]
    vitory_card = env_state[CARD_EFFECT_1_PLAYER_INDEX+CARD_EFFECT_PLAYER_LEN-1:CARD_EFFECT_4_PLAYER_INDEX+CARD_EFFECT_PLAYER_LEN+1:ALL_INFOR_PLAYER]
    score_real = score+vitory_card
    max_score = np.max(score_real)
    if max_score > 9:
        return False
    else:
        return True
    
def action_player(env_state,list_player,temp_file,per_file):
    current_player = int(env_state[ID_ACTION])
    player_state = getAgentState(env_state)
    played_move,temp_file[current_player],per_file = list_player[current_player](player_state,temp_file[current_player],per_file)
    return played_move,temp_file,per_file

@njit()
def getAgentSize():
    return 4

@njit()
def getActionSize():
    return 108

@njit()
def initEnv():
    env_state = np.zeros(INDEX)
    env_state[POINT_INDEX:ROBBER_BLOCK_INDEX] = np.full(POINT_LEN + ROAD_LEN, -1)
    env_state[TOKEN_BLOCK_INDEX : OFFER_MAIN_INDEX] = create_board()
    number_on_block = env_state[TOKEN_BLOCK_INDEX : TYPE_SOURCE_INDEX]
    index_where_robber = np.argmin(number_on_block)
    env_state[ROBBER_BLOCK_INDEX] = index_where_robber
    env_state[OTHER_IN4_INDEX] = -1 # Khi bắt đầu thì chưa có điểm đặt đường đầu tiên
    env_state[P1_SETTLEMENT_1ST:END_INDEX_FIRST_SETTLEMENT] = -1
    env_state[LARGEST_ARMY_PLAYER] = -1 #bắt đầu thì chưa có ai có quân đội mạnh nhất
    env_state[LONGEST_ROAD_PLAYER] = -1 #bắt đầu thì chưa cso ai có con đường dài nhất
    env_state[PLAYER_CAN_USE_DEV_CARD] = 0 #bắt đầu thì ko có khả năng dùng dev_card
    env_state[NUMBER_TRADE_OF_PLAYER] = 1 #bắt đầu thì cho trade tối đa 3 lần mỗi lượt
    env_state[CHECK_ROLL_INDEX] = 1
    env_state[PHASE] = 3
    return env_state

@njit()
def create_board():
    #list các ô kề với ô có index tương ứng
    list_set_block = [[1, 17, 12, 11],[0, 2, 17],[1, 3, 16, 17],[2, 4, 16],[3, 5, 15, 16],
                [4, 6, 15],[5, 7, 14, 15],[6, 8, 14],[7, 9, 13, 14],[8, 10, 13],
                [9, 11, 12, 13],[0, 10, 12],[0, 10, 11, 13, 17, 18],[8, 9, 10, 12, 14, 18],[6, 7, 8, 13, 15, 18],
                [4, 5, 6, 14, 16, 18],[2, 3, 4, 15, 17, 18],[0, 1, 2, 12, 16, 18],[12, 13, 14, 15, 16, 17]]
    all_block = np.arange(19)
    list_index = np.zeros(4)
    block_expect = np.array([0])
    for i in range(4):
        choice = int(np.random.choice(all_block))
        if len(block_expect) == 1:
            block_expect = np.concatenate((np.array(list_set_block[choice]), np.array([choice])))
        else:
            block_expect = np.concatenate((block_expect, np.array(list_set_block[choice]), np.array([choice])))
        list_index[i] =  choice
        temp_all_block = np.array([-1])
        for number in all_block:
            if number not in block_expect:
                if temp_all_block[0] == -1:
                    temp_all_block = np.array([number])
                else:
                    temp_all_block = np.append(temp_all_block, number)
        all_block = temp_all_block
    number_to_roll = np.array([11,12,9,4,5,10,3,11,4,10,9,0,3,5,2])
    np.random.shuffle(number_to_roll)
    list_index = np.sort(list_index)
    for i in range(len(list_index)):
        number_to_roll = np.concatenate((number_to_roll[:int(list_index[i])], np.array([(i+1)%2*8 + ((i+1)%2 == 0)*6]), number_to_roll[int(list_index[i]):]))
    all_type_source = np.concatenate((np.array([0]), np.full(4,1), np.full(3,2), np.full(4,3), np.full(4,4), np.full(3,5)))
    np.random.shuffle(all_type_source)
    desert_block = np.where(number_to_roll == 0)[0][0]
    desert_type = np.where(all_type_source == 0)[0][0]
    all_type_source[desert_block], all_type_source[desert_type] = all_type_source[desert_type], all_type_source[desert_block]
    
    harbor_deck = np.array([0,0,0,0,1,2,3,4,5])
    np.random.shuffle(harbor_deck)

    card_bank = np.full(5, 19)

    development_card = np.concatenate((np.zeros(14), np.array([1,1,2,2,3,3,4,4,4,4,4])))
    np.random.shuffle(development_card)

    return_in4 = np.concatenate((number_to_roll, all_type_source, harbor_deck, card_bank, development_card))
    return return_in4

@njit()
def getAgentState(env_state):
    id_action = int(env_state[ID_ACTION])
    player_state = env_state[ALL_INFOR_PLAYER*id_action:ALL_INFOR_PLAYER*(id_action+1)]
    for i in range(1, 4):
        player_state = np.append(player_state, env_state[ALL_INFOR_PLAYER*((i+id_action)%4):ALL_INFOR_PLAYER*((i+id_action)%4 + 1)][:ATTRIBUTE_PLAYER])
    map_owwner = env_state[POINT_INDEX : ROBBER_BLOCK_INDEX]
    map_owwner = np.where(map_owwner != -1, (map_owwner - id_action)%4 + 4*(map_owwner>3), map_owwner)
    source_in_bank = env_state[CARD_BANK_INDEX : CARD_BANK_INDEX+ CARD_BANK_LEN] > 0
    dev_card_in_bank = np.array([int(env_state[CARD_EFFECT_BANK_INDEX] != -1)])
    all_deal = env_state[OFFER_MAIN_INDEX : OFFER_MAIN_INDEX + OFFER_LEN*4]
    if id_action == env_state[MAIN_PLAYER] or env_state[PHASE] not in np.array([12, 13, 14, 15]):
        player_state = np.concatenate((player_state, map_owwner, env_state[ROBBER_BLOCK_INDEX:CARD_BANK_INDEX], source_in_bank, dev_card_in_bank, env_state[OFFER_MAIN_INDEX:OFFER_MAIN_INDEX + OFFER_LEN*4], env_state[OTHER_IN4_INDEX:TURN+1]))
    else:
        main_deal = all_deal[:OFFER_LEN]
        id_player_deal = int((id_action - env_state[MAIN_PLAYER])%4)
        own_deal = all_deal[OFFER_LEN*id_player_deal : OFFER_LEN*(id_player_deal+1)]
        player_state = np.concatenate((player_state, map_owwner, env_state[ROBBER_BLOCK_INDEX:CARD_BANK_INDEX], 
                                        source_in_bank, dev_card_in_bank, 
                                        main_deal[CARD_BANK_LEN:], main_deal[:CARD_BANK_LEN], 
                                        own_deal[CARD_BANK_LEN:], own_deal[:CARD_BANK_LEN], np.zeros(OFFER_LEN*2),
                                        env_state[OTHER_IN4_INDEX:TURN+1]))
    
    player_state[P_MAIN_PLAYER] = (player_state[P_MAIN_PLAYER] - player_state[P_ID_ACTION])%4
    player_state[P_ID_ACTION] = 0
    return player_state



@njit()
def find_max(length:int, diem_xp, list_road_da_di, p_road, unique_point):
    # print(list_road_da_di)

    list_road_check = [9999]
    for road in POINT_ROAD_RELATIVE[diem_xp]:
        if road != -1 and road in p_road:
            if road not in list_road_da_di:
                list_road_check.append(road)
    
    list_road_check.remove(9999)
    if len(list_road_check) == 0 or diem_xp not in unique_point:
        if length != 0:
            return length
    
    # print(list_road_check)
    list_length = [9999]
    for road in list_road_check:
        list_road_da_di_copy = list_road_da_di.copy()
        list_road_da_di_copy = np.append(list_road_da_di_copy, road)
        # print(list_road_da_di_copy+1)
        diem_xp_1 = 9999
        for diem in ROAD_BY_POINT[road]:
            if (diem != diem_xp 
                and diem != -1):
                diem_xp_1 = diem
                break
        if diem_xp_1 != 9999:
            list_length.append(find_max(length+1, diem_xp_1, list_road_da_di_copy, p_road, unique_point))
    list_length.remove(9999)
    return max(list_length)

@njit()
def calculator_longest_road(p_road, id_action, house_data):
    # print('xxxxxxx')
    house_data = house_data.astype(np.int64)
    p_road = p_road.astype(np.int64)
    point_of_road = ROAD_BY_POINT[p_road]
    unique_point_player = np.unique(point_of_road.flatten()).astype(np.int64)
    list_point_of_other_player = list_point_other_player(id_action, house_data)
    unique_point = np.array([-999])
    if len(list_point_of_other_player) > 0:
        unique_point = remote_a_if_a_in_b(unique_point_player, np.array(list_point_of_other_player))
    
    list_len_road = [-9999]
    for point in unique_point_player:
        list_len_road.append(find_max(0, point, np.array([-999]), p_road, unique_point))
    

    return max(list_len_road)

