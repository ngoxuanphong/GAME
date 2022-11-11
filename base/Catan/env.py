import pandas as pd
import numpy as np
import re
import numba
from numba import njit
import os
import time
from base.Catan.index import *
from base.Catan.basic_action import *
import random

@njit()
def getValidActions(player_state_origin):
    player_state = player_state_origin.copy()
    phase_env = int(player_state[P_PHASE])
    # phase_env = 2
    list_action = np.array([-1])
    turn = player_state[P_TURN]
    # print('turn', turn)
    roads_data = player_state[P_ROAD_INDEX : P_ROAD_INDEX+ROAD_LEN]
    points_data = player_state[P_POINT_INDEX : P_POINT_INDEX+POINT_LEN]
    player_source = player_state[P_CARD_PLAYER_INDEX:P_CARD_PLAYER_INDEX+CARD_PLAYER_LEN]
    if phase_env == 1:
        #đổ xúc sắc hay dùng thẻ dev nào?
        dev_card = player_state[P_DEV_CARD:P_DEV_CARD+4]
        list_action = np.array([102])
        #nếu được dùng dev_Card thì mới xét
        if player_state[P_PLAYER_CAN_USE_DEV_CARD] == 1:
            action_dev_card = np.where(dev_card>0)[0]
            if len(action_dev_card) > 0:
                list_action = np.append(list_action, action_dev_card+94)
                if 96 in list_action and np.sum(player_state[P_SOURCE_IN_BANK_INDEX : P_SOURCE_IN_BANK_INDEX+P_SOURCE_IN_BANK_LEN]) == 0:
                    list_action = np.delete(list_action, np.where(list_action == 96)[0])
                if 95 in list_action:
                    road_not_yet_have = np.where(roads_data == -1)[0]
                    p_roads = np.where(roads_data == 0)[0]
                    road_build, point_can_build = RoadCanBuild(p_roads, road_not_yet_have, 0, points_data)
                    if len(np.where(roads_data == 0)[0]) >= 15 or len(point_can_build) == 0:
                        list_action = np.delete(list_action, np.where(list_action == 95)[0])
        return list_action.astype(np.int64)
    elif phase_env == 2:
        '''
        1: Xây nhà,
        2: Nâng cấp city,
        3: Đặt đường,
        4: Mua thẻ effect
        '''
        #Xây nhà
        if CheckMaterialForAction(player_source,1):
            relationship = HouseCanBuild(points_data, roads_data, turn)
            if len(np.where(relationship > 0)[0]) > 0:
                if len(np.where(points_data == 0)[0]) < 5:
                    list_action = np.append(list_action,np.array([98]))
        #Nâng cấp lên City
        if CheckMaterialForAction(player_source,2):
            if len(np.where(points_data == 4)[0]) < 4 and len(np.where(points_data == 0)[0]) > 0:
                list_action = np.append(list_action,np.array([99]))
        #Đặt đường
        if CheckMaterialForAction(player_source,3):
            road_not_yet_have = np.where(roads_data == -1)[0]
            p_roads = np.where(roads_data == 0)[0]
            if len(road_not_yet_have) > 0 and len(p_roads) > 0:
                # RoadCanBuild(road_have, road_not_yet_have, id_action, points_data)
                road_build, point_can_build = RoadCanBuild(p_roads, road_not_yet_have, 0, points_data)
                if len(point_can_build) > 0 and len(p_roads) < 15:
                    list_action = np.append(list_action,np.array([100]))
        #Mua thẻ dev
        dev_card = player_state[P_DEV_CARD:P_DEV_CARD+4]
        if CheckMaterialForAction(player_source,4) and len(np.where(dev_card>-1)[0]) != 0:
            if player_state[P_DEV_CARD_IN_BANK_INDEX] == 1:
                list_action = np.append(list_action,np.array([101]))
        # Dùng thẻ dev
        if player_state[P_PLAYER_CAN_USE_DEV_CARD] == 1:
            list_action = np.append(list_action, np.where(dev_card>0)[0]+94)
            if 96 in list_action and np.sum(player_state[P_SOURCE_IN_BANK_INDEX : P_SOURCE_IN_BANK_INDEX+P_SOURCE_IN_BANK_LEN]) == 0:
                list_action = np.delete(list_action, np.where(list_action == 96)[0])
            if 95 in list_action and len(np.where(roads_data == 0)[0]) >= 15:
                list_action = np.delete(list_action, np.where(list_action == 95)[0])
        # Trading tài Nguyên
        if player_state[P_NUMBER_TRADE_OF_PLAYER] > 0 and np.sum(player_source) > 0:
            max_source_in_deal = np.max(np.array([player_state[P_P1_ATTRIBUTE_PLAYER+1], player_state[P_P2_ATTRIBUTE_PLAYER+1], player_state[P_P3_ATTRIBUTE_PLAYER+1]]))
            if max_source_in_deal > 0:
                list_action = np.append(list_action, np.array([106]))       #trading vs người
        #Trading vs port and bank
        player_res_trade_bank = np.where(player_source>3)[0]
        bank_res = np.where(player_state[P_SOURCE_IN_BANK_INDEX:P_SOURCE_IN_BANK_INDEX+P_SOURCE_IN_BANK_LEN] > 0)[0]
        if len(player_res_trade_bank)*len(bank_res) > 0:
            if len(bank_res) > 1 or len(player_res_trade_bank) > 1 or player_res_trade_bank[0] != bank_res[0]:
                # print('toang0', len(bank_res), len(player_res_trade_bank), player_res_trade_bank[0], bank_res[0])
                list_action = np.append(list_action,np.array([107]))
            # else:
            #     if len(player_res_trade_bank) > 1 or player_res_trade_bank[0] != bank_res[0]:
            #         list_action = np.append(list_action,np.array([107]))
        if list_action[-1] != 107 and len(bank_res) > 0:
            type_of_port = player_state[P_TYPE_HARBOR_INDEX : P_TYPE_HARBOR_INDEX+TYPE_HARBOR_LEN]  #Loại cảng theo thứ tự
            point_in_port = points_data[POINT_IN_HARBOR]                #xét các điểm là cảng
            # print(player_source,'-----', type_of_port, point_in_port)
            player_port = np.where((point_in_port == 0) | (point_in_port == 4))[0]     #tìm các điểm là cảng của người chơi
            if len(player_port) > 0:
                for id in player_port:
                    type_port = type_of_port[id//2]
                    if type_port == 0:
                        if np.max(player_source) > 2:
                            if len(bank_res) > 1 or bank_res[0] != np.argmax(player_source):
                                # print('toang1', len(bank_res), bank_res[0], np.argmax(player_source))
                                list_action = np.append(list_action,np.array([107]))
                            break
                    elif type_port != 0 and player_source[int(type_port-1)] > 1:
                        if len(bank_res) > 1 or bank_res[0] != player_source[int(type_port-1)]:
                            # print('toang2', len(bank_res), bank_res[0], player_source[int(type_port-1)])
                            list_action = np.append(list_action,np.array([107]))
                        break
                    else:
                        pass
        list_action = np.append(list_action,np.array([103]))
        list_action = np.delete(list_action,0)       #bỏ qua -1 ở đầu 
        return list_action.astype(np.int64)
    elif phase_env == 3:
        list_action = HouseCanBuild(points_data, roads_data, turn)
        return list_action
    elif phase_env == 4:#upgrade city (các điểm upgrade được)
        point_can_upgrade_city = np.where(points_data == 0)[0]
        return point_can_upgrade_city.astype(np.int64)
    elif phase_env == 5: #chọn đỉnh thứ nhất của đường khi đặt đường hoặc dùng buildroad
        road_not_yet_have = np.where(roads_data == -1)[0]
        road_have = np.where(roads_data == 0)[0]
        # RoadCanBuild(road_have, road_not_yet_have, 0, points_data)
        road_build, list_action = RoadCanBuild(road_have, road_not_yet_have, 0, points_data)
        if len(list_action) == 0 or len(road_have) == 15:
            list_action = np.array([103])
        return list_action

    elif phase_env == 6: #chọn đỉnh thứ 2 của đường
        road_not_yet_have = np.where(roads_data == -1)[0]
        road_have = np.where(roads_data == 0)[0]
        
        road_build, list_action = RoadCanBuild(road_have, road_not_yet_have, 0, points_data)
        temp_point = player_state[P_TEMP_POINT]
        road_have_point = np.where(ROAD_BY_POINT == temp_point)[0]
        road_have_point_check = []
        if turn < 8:
            for road in road_have_point:
                if player_state[P_ROAD_INDEX+road] == -1:
                    road_have_point_check.append(road)
        else:
            for road in road_have_point:
                if road in road_build:
                    road_have_point_check.append(road)
        list_point = ROAD_BY_POINT[np.array(road_have_point_check)].flatten()
        list_action = list_point[list_point != temp_point]
        return list_action.astype(np.int64)
    elif phase_env == 7:
        #chọn ô đặt rober khi đổ 7 hoặc dùng knight 
        all_block_number = player_state[P_TOKEN_BLOCK_INDEX:P_TYPE_SOURCE_INDEX]
        robber_block = int(player_state[P_ROBBER_BLOCK_INDEX])
        all_block_number[robber_block] = -1
        list_action = np.where(all_block_number > -1)[0] + 54
        return list_action.astype(np.int64)
    elif phase_env == 8:
        #chọn tài nguyên để bỏ đi khi bị chia bài
        list_action = np.where(player_state[P_CARD_PLAYER_INDEX:P_DEV_CARD] > 0)[0] + 73
        return list_action.astype(np.int64)
    elif phase_env == 9:
        #chọn người để cướp
        robber_block = int(player_state[P_ROBBER_BLOCK_INDEX])
        point_in_block = player_state[P_POINT_INDEX:P_ROAD_INDEX][POINT_IN_BLOCK[robber_block]]
        point_in_block = point_in_block[point_in_block > -1]            #chỉ xét các đỉnh đã bị sở hữu
        point_in_block = point_in_block % 4

        point_in_block = point_in_block[point_in_block != 0]            #loại bản thân khỏi danh sách cướp
        player_can_steal = np.unique(point_in_block)
        for player in player_can_steal:
            if player_state[int(P_P1_ATTRIBUTE_PLAYER + (player-1)*P_ATTRIBUTE_PLAYER_LEN+1)] > 0:
                list_action = np.append(list_action, int(player))
        list_action = np.delete(list_action, np.where(list_action == -1)[0])
        list_action = (list_action + 77).astype(np.int64)
        return list_action
    elif phase_env == 10:
        #chọn tài nguyên khi dùng year_of_plenty, chỉ chọn những cái mà ngân hàng còn
        source_card_in_bank = player_state[P_SOURCE_IN_BANK_INDEX:P_SOURCE_IN_BANK_INDEX+P_SOURCE_IN_BANK_LEN]
        list_action = np.where(source_card_in_bank != 0)[0] + 86
        return list_action.astype(np.int64)
    elif phase_env == 11:
        #chọn tài nguyên khi dùng monopoly
        list_action = np.arange(86, 91)
        return list_action.astype(np.int64)
    elif phase_env == 12:
        # print('tai_nguyen: ', player_source)
        #chọn tài nguyên để trading (nếu đã có tài nguyên rồi thì phải có action dừng)
        #chỉ nạp các tài nguyên người chơi có
        if player_state[P_ID_ACTION] == player_state[P_MAIN_PLAYER]:
            # print('NGON NGHẺ', player_state[P_ID_ACTION], player_state[P_MAIN_PLAYER])
            main_deal = player_state[P_OFFER_MAIN_INDEX:P_OFFER_MAIN_INDEX + OFFER_LEN]
            # print(player_source,'main_deal: ', main_deal)
            if len(np.where(main_deal[:CARD_BANK_LEN] > 0)[0]) > 3:     #neu da co 4 loai tai nguyen trong trade bang cach gan cho no bang 0 trong playersource
                player_source[np.where(main_deal[:CARD_BANK_LEN] == 0)[0][0]] = 0

            list_action = np.where(player_source > main_deal[:CARD_BANK_LEN])[0] + 81
            if np.sum(main_deal) > 0:
                list_action = np.append(list_action, 103)
        else:
            other_deal = player_state[P_OFFER_1_INDEX:P_OFFER_1_INDEX + OFFER_LEN]
            if len(np.where(player_source > other_deal[:CARD_BANK_LEN])[0]) > 3:     #neu da co 4 loai tai nguyen trong trade bang cach gan cho no bang 0 trong playersource
                player_source[np.where(other_deal[:CARD_BANK_LEN] == 0)[0][0]] = 0

            list_action = np.where(player_source > other_deal[:CARD_BANK_LEN])[0] + 81
            if np.sum(other_deal) > 0:
                list_action = np.append(list_action, 103)
        return list_action.astype(np.int64)
    elif phase_env == 13:
        #chọn tài nguyên muốn nhận được (nếu đã có tài nguyên rồi thì phải có action dừng)
        own_offer = np.zeros(10)
        max_source_in_deal = 0
        if player_state[P_ID_ACTION] == player_state[P_MAIN_PLAYER]:
            own_offer = player_state[P_OFFER_MAIN_INDEX:P_OFFER_MAIN_INDEX+OFFER_LEN]
            max_source_in_deal = np.max(np.array([player_state[P_P1_ATTRIBUTE_PLAYER+1], player_state[P_P2_ATTRIBUTE_PLAYER+1], player_state[P_P3_ATTRIBUTE_PLAYER+1]]))
        # else:
        #     #người chơi khác đổi tối đa chỉ bằng số thẻ của người chơi chính
        #     own_offer = player_state[P_OFFER_1_INDEX:P_OFFER_1_INDEX+OFFER_LEN]
        #     max_source_in_deal = player_state[int(P_P1_ATTRIBUTE_PLAYER+P_ATTRIBUTE_PLAYER_LEN*((player_state[P_ID_ACTION] - player_state[P_MAIN_PLAYER])%4-1))]
        

        list_action = np.where(own_offer[:5] == 0)[0] + 86          #chi nap cac tai nguyen minh k bo ra
        
        if np.sum(own_offer[5:]) > 0 and np.sum(own_offer[5:]) < max_source_in_deal:
            list_action = np.append(list_action, 103)
        elif np.sum(own_offer[5:]) >= max_source_in_deal:
            list_action = np.array([103])
        return list_action.astype(np.int64)
    elif phase_env == 14:
        #chọn đổi với ai
        list_action = np.array([104])       #luôn có action ko trao đổi vs ai
        all_trade = player_state[P_OFFER_1_INDEX:P_OFFER_1_INDEX+OFFER_LEN*3]
        trade_source = np.zeros(3)
        for i in range(3):
            other_offer = all_trade[OFFER_LEN*i:OFFER_LEN*(i+1)][:CARD_BANK_LEN]
            if np.sum(other_offer) > 0 and np.sum(other_offer > player_source) == 0:
                trade_source[i] = 1
        list_action = np.append(list_action, np.where(trade_source > 0)[0] + 91)
        list_action = list_action.astype(np.int64)
        return list_action
    elif phase_env == 15:
        #người chơi phụ nhận deal từ người chơi chính, đồng ý hoặc ko hoặc sửa deal
        list_action = np.array([104])
        main_deal = player_state[P_OFFER_MAIN_INDEX:P_OFFER_MAIN_INDEX + OFFER_LEN][:CARD_BANK_LEN]
        #nếu đủ tài  nguyên cho deal thì thêm cái action đồng ý
        if np.sum(main_deal > player_source) == 0:
            # print(main_deal, player_source, 'ashdashdahsdgas')
            list_action = np.append(list_action, 105)
            #nếu ko đồng ý và có thể deal thì nạp nguyên liệu luôn vào deal
            # list_action = np.append(list_action, np.where(player_source > 0)[0] + 81)         #update tắt redeal 08/09/2022
        return list_action.astype(np.int64)
    elif phase_env == 16:
        #người chơi chọn tài nguyên bỏ ra để đổi vs cảng hoặc ngân hàng
        trade_bank = np.where(player_source > 3)[0]
        # print('checkbank16', trade_bank)
        if len(trade_bank) > 0:
            list_action = np.append(list_action, trade_bank)
        type_of_port = player_state[P_TYPE_HARBOR_INDEX : P_TYPE_HARBOR_INDEX+TYPE_HARBOR_LEN]  #Loại cảng theo thứ tự
        point_in_port = points_data[POINT_IN_HARBOR]                #xét các điểm là cảng
        player_port = np.where((point_in_port == 0) | (point_in_port == 4))[0]     #tìm các điểm là cảng của người chơi
        if len(player_port) > 0:
            for id in player_port:
                type_port = type_of_port[id//2]
                if type_port == 0:
                    trade_port_3 = np.where(player_source > 2)[0]
                    if len(trade_port_3) > 0:
                        list_action = np.append(list_action, trade_port_3)
                elif type_port != 0 and player_source[int(type_port-1)] > 1:
                    list_action = np.append(list_action,np.array([int(type_port-1)]))
        bank_res = np.where(player_state[P_SOURCE_IN_BANK_INDEX:P_SOURCE_IN_BANK_INDEX+P_SOURCE_IN_BANK_LEN] > 0)[0]
        if len(bank_res) == 1 and bank_res[0] in list_action:
            #chống không cho đổi loại tài nguyên ngân hàng có duy nhất
            list_action = np.delete(list_action, np.where(list_action == bank_res[0])[0])
        list_action = list_action[1:] + 81       #bỏ qua -1 ở đầu 
        return list_action.astype(np.int64)
    elif phase_env == 17:
        #người chơi chọn lấy tài nguyên mà ngân hàng có nhưng ko phải cái mình bỏ ra
        main_deal = player_state[P_OFFER_MAIN_INDEX:P_OFFER_MAIN_INDEX + OFFER_LEN]
        resource_sell = np.argmax(main_deal)
        source_card_in_bank = player_state[P_SOURCE_IN_BANK_INDEX:P_SOURCE_IN_BANK_INDEX+P_SOURCE_IN_BANK_LEN]
        source_card_in_bank[resource_sell] = 0
        list_action = np.where(source_card_in_bank != 0)[0] + 86

        return list_action.astype(np.int64)
   
@njit()
def stepEnv(env_state, action):
    id_action = int(env_state[ID_ACTION])
    turn = int(env_state[TURN])
    phase_env = env_state[PHASE]
    house_data = env_state[POINT_INDEX: POINT_INDEX+POINT_LEN]
    road_data = env_state[ROAD_INDEX:ROAD_INDEX+ROAD_LEN]
    if phase_env == 1:
        '''nếu đổ xúc sắc thì roll rồi cộng tài nguyên cho các người chơi khác, nếu roll ra 7
        thì check xem có cần chia bài ko, nếu chia thì về phase8, còn ko chia thì về phase7, ko 
        đổ ra 7 thì lên phase2 
        nếu dùng thẻ thì chuyển về phase tương ứng (5,7,10,11)
        '''
        if action == 102:
            #nếu đổ xúc sắc
            dice1 = np.random.randint(1,7)
            dice2 = np.random.randint(1,7)
           
            env_state[LAST_ROLL] = dice1 + dice2
            # print('xúc sắc ra ', env_state[LAST_ROLL] )
            # print('all_last_in4_deal', env_state[OFFER_MAIN_INDEX:OFFER_MAIN_INDEX+40])
            # print('ĐỔ xúc sắc ra ', env_state[LAST_ROLL])
            env_state[CHECK_ROLL_INDEX] = 0
            if env_state[LAST_ROLL] == 7:
                #đi chia bài
                # env_state[MAIN_PLAYER] = id_action
                if np.sum(env_state[ALL_INFOR_PLAYER*id_action:ALL_INFOR_PLAYER*(id_action+1)][CARD_1_PLAYER_INDEX:CARD_1_PLAYER_INDEX+CARD_PLAYER_LEN]) > 7:
                    env_state[PHASE] = 8
                    env_state[CARD_NEED_DROP] = np.sum(env_state[ALL_INFOR_PLAYER*id_action:ALL_INFOR_PLAYER*(id_action+1)][CARD_1_PLAYER_INDEX:CARD_1_PLAYER_INDEX+CARD_PLAYER_LEN]) // 2
                else:
                    env_state[PHASE] = 7 
                    #nếu ko ai phải chia bài, nhảy đến phase 7 để di knight
                    for id in range(1,4):
                        id_next = (id_action + id) % 4
                        if np.sum(env_state[ALL_INFOR_PLAYER*id_next:ALL_INFOR_PLAYER*(id_next+1)][CARD_1_PLAYER_INDEX:CARD_1_PLAYER_INDEX+CARD_PLAYER_LEN]) > 7:
                            env_state[PHASE] = 8
                            env_state[ID_ACTION] = id_next
                            env_state[CARD_NEED_DROP] = np.sum(env_state[ALL_INFOR_PLAYER*id_next:ALL_INFOR_PLAYER*(id_next+1)][CARD_1_PLAYER_INDEX:CARD_1_PLAYER_INDEX+CARD_PLAYER_LEN]) // 2
                            break
                    
            else:
                #nếu ko phải di knight thì nhận tài nguyên rồi về phase 2
                # print(env_state)
                a = env_state.copy()
                env_state = process_return_roll(env_state)
                # for i in range(4):
                #     print(env_state[int(i*ALL_INFOR_PLAYER+5): int(i*ALL_INFOR_PLAYER+10)])
                env_state[PHASE] = 2
        else:
            env_state[PLAYER_CAN_USE_DEV_CARD] = 0
            env_state[int(ALL_INFOR_PLAYER*id_action + 2)] -= 1         #giảm số lượng thẻ effect đang ẩn
            env_state[ALL_INFOR_PLAYER*id_action + CARD_EFFECT_1_PLAYER_INDEX + action - 94] -= 1
            if action == 94:
                env_state = dung_the_knight(env_state)
                env_state[PHASE] = 7
            elif action == 95:
                env_state[PHASE] = 5
                env_state[USE_BUILD_ROAD] = 2
            elif action == 96:
                env_state[PHASE] = 10
                env_state[USE_YEAR_OF_PLENTY] = 1
            else:
                env_state[PHASE] = 11
        return env_state
    elif phase_env == 2:
        #nếu xây nhà thì chuyển phase3, upgrade thì chuyển phase4, đặt đường thì sang phas5
        # print('thong tin ng choi: ', env_state[ALL_INFOR_PLAYER*id_action:ALL_INFOR_PLAYER*(id_action+1)])
        if action == 98:
            env_state[PHASE] = 3        
        elif action == 99:
            env_state[PHASE] = 4
        elif action == 100:
            env_state[PHASE] = 5
        #dùng knight thì sang phase7, year_plenty thì phase10, monopoly thì phase 11
        elif action == 94:
            env_state = dung_the_knight(env_state)
            env_state[PLAYER_CAN_USE_DEV_CARD] = 0
            env_state[PHASE] = 7
            env_state[ALL_INFOR_PLAYER*id_action + CARD_EFFECT_1_PLAYER_INDEX + 0] -= 1
            env_state[int(ALL_INFOR_PLAYER*id_action + 2)] -= 1         #giảm số lượng thẻ effect đang ẩn
        elif action == 96:
            env_state[PHASE] = 10
            env_state[USE_YEAR_OF_PLENTY] = 1
            env_state[PLAYER_CAN_USE_DEV_CARD] = 0
            env_state[ALL_INFOR_PLAYER*id_action + CARD_EFFECT_1_PLAYER_INDEX + 2] -= 1
            env_state[int(ALL_INFOR_PLAYER*id_action + 2)] -= 1         #giảm số lượng thẻ effect đang ẩn

        elif action == 97:
            env_state[PHASE] = 11
            env_state[PLAYER_CAN_USE_DEV_CARD] = 0
            env_state[ALL_INFOR_PLAYER*id_action + CARD_EFFECT_1_PLAYER_INDEX + 3] -= 1
            env_state[int(ALL_INFOR_PLAYER*id_action + 2)] -= 1         #giảm số lượng thẻ effect đang ẩn

        elif action == 95:
            env_state[PHASE] = 5
            env_state[USE_BUILD_ROAD] = 2
            env_state[PLAYER_CAN_USE_DEV_CARD] = 0
            env_state[ALL_INFOR_PLAYER*id_action + CARD_EFFECT_1_PLAYER_INDEX + 1] -= 1
            env_state[int(ALL_INFOR_PLAYER*id_action + 2)] -= 1         #giảm số lượng thẻ effect đang ẩn

        #nếu mua dev card thì mua rồi lại lặp phase2
        elif action == 101:
            env_state[ALL_INFOR_PLAYER*id_action + CARD_1_PLAYER_INDEX:ALL_INFOR_PLAYER*id_action + CARD_1_PLAYER_INDEX+ CARD_BANK_LEN] = env_state[ALL_INFOR_PLAYER*id_action + CARD_1_PLAYER_INDEX:ALL_INFOR_PLAYER*id_action + CARD_1_PLAYER_INDEX+ CARD_BANK_LEN] - np.array([0,0,1,1,1])
            env_state[CARD_BANK_INDEX : CARD_BANK_INDEX + CARD_BANK_LEN] = env_state[CARD_BANK_INDEX : CARD_BANK_INDEX + CARD_BANK_LEN]  + np.array([0,0,1,1,1])
            card = env_state[CARD_EFFECT_BANK_INDEX]
            env_state[int(ALL_INFOR_PLAYER*id_action + CARD_EFFECT_1_PLAYER_INDEX+card)] += 1
            env_state[int(ALL_INFOR_PLAYER*id_action + 2)] += 1
            env_state[CARD_EFFECT_BANK_INDEX:CARD_EFFECT_BANK_INDEX+CARD_EFFECT_BANK_LEN] = np.concatenate((env_state[CARD_EFFECT_BANK_INDEX+1:CARD_EFFECT_BANK_INDEX+CARD_EFFECT_BANK_LEN],np.array([-1])))
            env_state[PHASE] = 2
            env_state = updateSumMaterial(env_state)
        #nếu trading với người vào phase12, dùng devcard thì tham khảo phase1
        elif action == 106:
            env_state[NUMBER_TRADE_OF_PLAYER] -= 1
            env_state[PHASE] = 12
        #nếu trading cảng thì sang phase 16
        elif action == 107:
            player_source = env_state[int(ALL_INFOR_PLAYER*id_action+ATTRIBUTE_PLAYER) : int(ALL_INFOR_PLAYER*id_action+ATTRIBUTE_PLAYER+CARD_PLAYER_LEN)]
            house_data = env_state[POINT_INDEX: POINT_INDEX+POINT_LEN]
            point_in_port = house_data[POINT_IN_HARBOR]
            player_port = np.where((point_in_port == id_action) | (point_in_port == (id_action + 4)))[0]     #tìm các điểm là cảng của người chơi
            type_of_port = env_state[TYPE_HARBOR_INDEX : TYPE_HARBOR_INDEX+TYPE_HARBOR_LEN]
            # print(player_source,'cảng', player_port, type_of_port)
            env_state[PHASE] = 16
        #nếu endturn thì check xem người kế có dev_card ko, nếu có thì về phase 1, chuyển ng chơi, nếu ko thì roll luôn
        elif action == 103:
            env_state[ID_ACTION] = (id_action+1)%4
            env_state[MAIN_PLAYER] = env_state[ID_ACTION]
            env_state[PLAYER_CAN_USE_DEV_CARD] = np.sum(env_state[int(ALL_INFOR_PLAYER*env_state[ID_ACTION]+CARD_EFFECT_1_PLAYER_INDEX):int(ALL_INFOR_PLAYER*env_state[ID_ACTION]+CARD_EFFECT_1_PLAYER_INDEX+CARD_EFFECT_PLAYER_LEN -1)]) > 0
            if env_state[TURN] >= 8:
                env_state[PHASE] = 1
            env_state[TURN] += 1
            env_state[NUMBER_TRADE_OF_PLAYER] = 1
            # for i in range(4):
            #     print(env_state[int(i*ALL_INFOR_PLAYER+5): int(i*ALL_INFOR_PLAYER+10)])
            # print('all_deal', env_state[OFFER_MAIN_INDEX:OFFER_MAIN_INDEX + OFFER_LEN*4])
            env_state[CHECK_ROLL_INDEX] = 1
        return env_state
    elif phase_env == 3:
        #cập nhật nhà, trừ tài nguyên, về phase2
        env_state[POINT_INDEX + action] = id_action
        if turn > 8:
            #trừ tài nguyên xây nhà
            env_state[ALL_INFOR_PLAYER*id_action + CARD_1_PLAYER_INDEX:ALL_INFOR_PLAYER*id_action + CARD_1_PLAYER_INDEX+ CARD_BANK_LEN] = env_state[ALL_INFOR_PLAYER*id_action + CARD_1_PLAYER_INDEX:ALL_INFOR_PLAYER*id_action + CARD_1_PLAYER_INDEX+ CARD_BANK_LEN] - np.array([1,1,1,1,0])
            env_state[CARD_BANK_INDEX : CARD_BANK_INDEX + CARD_BANK_LEN] = env_state[CARD_BANK_INDEX : CARD_BANK_INDEX + CARD_BANK_LEN]  + np.array([1,1,1,1,0])
            env_state[PHASE] = 2

        elif turn >=4 and turn <8:
            all_type_source = env_state[TYPE_SOURCE_INDEX:TYPE_HARBOR_INDEX]
            list_token = GetMaterialFirst(all_type_source,action)
            env_state[ALL_INFOR_PLAYER*id_action + CARD_1_PLAYER_INDEX:ALL_INFOR_PLAYER*id_action + CARD_1_PLAYER_INDEX+ CARD_BANK_LEN] = env_state[ALL_INFOR_PLAYER*id_action + CARD_1_PLAYER_INDEX:ALL_INFOR_PLAYER*id_action + CARD_1_PLAYER_INDEX+ CARD_BANK_LEN] + list_token
            env_state[CARD_BANK_INDEX : CARD_BANK_INDEX + CARD_BANK_LEN] = env_state[CARD_BANK_INDEX : CARD_BANK_INDEX + CARD_BANK_LEN]  - list_token

        if turn < 8:
            env_state[PHASE] = 6
            env_state[OTHER_IN4_INDEX] = action
        else:
            env_state[PHASE] = 2
        env_state = updateSumMaterial(env_state)
        env_state[int(ALL_INFOR_PLAYER*id_action)] += 1 #Cộng thêm điểm cho người xây nhà
        # Check longest road

        road_near_point = POINT_ROAD_RELATIVE[action]
        road_near_point = road_near_point[road_near_point != -1]

        if len(road_near_point) >= 2:
            player_have_road_cut = -1
            if len(road_near_point) == 2:
                if road_data[road_near_point[0]] == road_data[road_near_point[1]] and road_data[road_near_point[1]] != -1:
                    player_have_road_cut = road_data[road_near_point[1]]
            if len(road_near_point) == 3:
                if road_data[road_near_point[0]] == road_data[road_near_point[1]] and road_data[road_near_point[1]] != -1:
                    player_have_road_cut = road_data[road_near_point[1]]
                if road_data[road_near_point[2]] == road_data[road_near_point[1]] and road_data[road_near_point[1]] != -1:
                    player_have_road_cut = road_data[road_near_point[1]]
                if road_data[road_near_point[0]] == road_data[road_near_point[2]] and road_data[road_near_point[2]] != -1:
                    player_have_road_cut = road_data[road_near_point[2]]

            if player_have_road_cut != -1:
                player_have_road_cut = int(player_have_road_cut)
                p_road = np.where(road_data == player_have_road_cut)[0].astype(np.int64)
                longest_road_player = env_state[LONGEST_ROAD_PLAYER]
                house_data = env_state[POINT_INDEX: int(POINT_INDEX+POINT_LEN)].astype(np.int64)
                if len(p_road) > 0:
                    # player_have_road_cut = 0
                    longest_road = calculator_longest_road(p_road, player_have_road_cut, house_data)
                    env_state[int(ALL_INFOR_PLAYER*player_have_road_cut + 4)] = longest_road
                    # longest_road_other_player = np.array([env_state[int(ALL_INFOR_PLAYER*i + 4)] for i in range(4)])
                    longest_road_other_player = env_state[ATTRIBUTE_PLAYER_1_INDEX:ATTRIBUTE_PLAYER_4_INDEX+4:ALL_INFOR_PLAYER]
                    if longest_road_player != -1 and longest_road_player == player_have_road_cut:
                        if longest_road < max(longest_road_other_player):
                            count_player_longest_road = len(np.where(longest_road_other_player == max(longest_road_other_player))[0])
                            if count_player_longest_road >= 1:
                                env_state[int(ALL_INFOR_PLAYER*longest_road_player)] -= 2
                                env_state[LONGEST_ROAD_PLAYER] = -1
                                if count_player_longest_road == 1:
                                    player_longest_road = np.argmax(longest_road_other_player)
                                    env_state[int(ALL_INFOR_PLAYER*player_longest_road)] += 2
                                    env_state[LONGEST_ROAD_PLAYER] = player_longest_road
        return env_state
    elif phase_env == 4:
        #cập nhật city, trừ tài nguyên, về phase2
        env_state[POINT_INDEX + action] = id_action + 4
        position_curson = ALL_INFOR_PLAYER*id_action
        env_state[ALL_INFOR_PLAYER*id_action + CARD_1_PLAYER_INDEX:ALL_INFOR_PLAYER*id_action + CARD_1_PLAYER_INDEX+ CARD_BANK_LEN] = env_state[ALL_INFOR_PLAYER*id_action + CARD_1_PLAYER_INDEX:ALL_INFOR_PLAYER*id_action + CARD_1_PLAYER_INDEX+ CARD_BANK_LEN] - np.array([0,0,0,2,3])
        env_state[CARD_BANK_INDEX : CARD_BANK_INDEX + CARD_BANK_LEN] = env_state[CARD_BANK_INDEX : CARD_BANK_INDEX + CARD_BANK_LEN]  + np.array([0,0,0,2,3])
        env_state[int(ALL_INFOR_PLAYER*id_action)] += 1
        env_state[PHASE] = 2
        env_state = updateSumMaterial(env_state)
        return env_state
    elif phase_env == 5: #cập nhật temp_point, sang phase 6
        if action == 103:
            env_state[PHASE] = 2
        else:
            env_state[OTHER_IN4_INDEX] = action #Update temp_point in env_state
            env_state[PHASE] = 6 #To phase 6
        return env_state
    elif phase_env == 6:
        temp_point = env_state[OTHER_IN4_INDEX]  #đỉnh thứ nhất của đường
        if turn > 7 and env_state[USE_BUILD_ROAD] == 0:
            env_state[ALL_INFOR_PLAYER*id_action + CARD_1_PLAYER_INDEX:ALL_INFOR_PLAYER*id_action + CARD_1_PLAYER_INDEX+ CARD_BANK_LEN] =  env_state[ALL_INFOR_PLAYER*id_action + CARD_1_PLAYER_INDEX:ALL_INFOR_PLAYER*id_action + CARD_1_PLAYER_INDEX+ CARD_BANK_LEN] - ROAD  #Trả nguyên liệu cho bàn chơi
            env_state[CARD_BANK_INDEX : CARD_BANK_INDEX + CARD_BANK_LEN] = env_state[CARD_BANK_INDEX : CARD_BANK_INDEX + CARD_BANK_LEN] + ROAD
            env_state[ATTRIBUTE_PLAYER_1_INDEX + 1] = np.sum(env_state[CARD_1_PLAYER_INDEX : CARD_1_PLAYER_INDEX + CARD_BANK_LEN])
            env_state[ATTRIBUTE_PLAYER_2_INDEX + 1] = np.sum(env_state[CARD_2_PLAYER_INDEX : CARD_2_PLAYER_INDEX + CARD_BANK_LEN])
            env_state[ATTRIBUTE_PLAYER_3_INDEX + 1] = np.sum(env_state[CARD_3_PLAYER_INDEX : CARD_3_PLAYER_INDEX + CARD_BANK_LEN])
            env_state[ATTRIBUTE_PLAYER_4_INDEX + 1] = np.sum(env_state[CARD_4_PLAYER_INDEX : CARD_4_PLAYER_INDEX + CARD_BANK_LEN])

        for road in np.where(ROAD_BY_POINT == temp_point)[0]: #update road_index in env_state
            if action in ROAD_BY_POINT[road]:
                env_state[ROAD_INDEX + road] = id_action


        p_road = np.where(env_state[ROAD_INDEX:ROAD_INDEX+ROAD_LEN] == id_action)[0] #đường của người choi sỏ hữu
        road_not_yet_have = np.where(env_state[ROAD_INDEX:ROAD_INDEX+ROAD_LEN] == -1)[0] #Đường chua có nguoif nào sở hữu
        points_data = env_state[POINT_INDEX: POINT_INDEX+POINT_LEN] #thông tin các điểm ở trên bàn
        
        road_build, list_action = RoadCanBuild(p_road, road_not_yet_have, id_action, points_data)

        if env_state[USE_BUILD_ROAD] != 0: #khi dùng thẻ road building
            env_state[USE_BUILD_ROAD] -= 1 
            env_state[PHASE] = 5
            if env_state[USE_BUILD_ROAD] == 0 or len(p_road) == 15 or len(list_action) == 0: #hết lượt dùng thẻ road, hoạc đủ 15 đường, hoặc không còn dduofng để xây
                env_state[PHASE] = 2
                env_state[USE_BUILD_ROAD] = 0
            if env_state[USE_BUILD_ROAD] == 0 and env_state[CHECK_ROLL_INDEX] == 1:
                env_state[PHASE] = 1
        else:
            if turn > 7:
                env_state[PHASE] = 2 #to phase 2
            elif turn < 3:
                env_state[PHASE] = 3
                env_state[ID_ACTION] = (env_state[ID_ACTION]+1)%4
                env_state[TURN] += 1
            elif turn >3 and turn < 7:
                env_state[TURN] += 1
                env_state[PHASE] = 3
                env_state[ID_ACTION] = (env_state[ID_ACTION]-1)%4
            elif turn == 3: 
                env_state[TURN] += 1
                env_state[PHASE] = 3
            elif turn == 7:
                env_state[TURN] += 1
                env_state[PHASE] = 1

        env_state[OTHER_IN4_INDEX] = -1

        # check longest road
        p_road = np.where(env_state[ROAD_INDEX:ROAD_INDEX+ROAD_LEN] == id_action)[0]
        longest_road_player = env_state[LONGEST_ROAD_PLAYER]
        if len(p_road) > 0:
            longest_road = calculator_longest_road(p_road, id_action, house_data)
            env_state[int(ALL_INFOR_PLAYER*id_action + 4)] = longest_road
            if longest_road_player == -1:
                if longest_road >= 5:
                    env_state[int(ALL_INFOR_PLAYER*id_action)] += 2
                    env_state[LONGEST_ROAD_PLAYER] = id_action
            else:
                if longest_road_player != id_action:
                    if longest_road > env_state[int(ALL_INFOR_PLAYER*longest_road_player + 4)]:
                        env_state[int(ALL_INFOR_PLAYER*id_action)] += 2
                        env_state[int(ALL_INFOR_PLAYER*longest_road_player)] -= 2
                        env_state[LONGEST_ROAD_PLAYER] = id_action
        # print(env_state[int(ALL_INFOR_PLAYER*id_action + 4)])
        return env_state
    elif phase_env == 7:
        #cập nhật vị trí robber theo env_state[ROBBER_BLOCK_INDEX], kiểm tra các người chơi còn tài nguyên hay ko để đi cướp
        block_robber = action - 54
        env_state[ROBBER_BLOCK_INDEX] = block_robber
        all_player_map = env_state[POINT_INDEX:POINT_INDEX+POINT_LEN]
        point_in_block = all_player_map[POINT_IN_BLOCK[block_robber]]
        point_in_block = point_in_block[(point_in_block > -1) & (point_in_block != id_action) & (point_in_block != (id_action+4))]      #xét các điểm bị sở hữu k phải của người chơi hiện tại, xem cướp đc k
        if env_state[CHECK_ROLL_INDEX] == 1:
            env_state[PHASE] = 1
        else:
            env_state[PHASE] = 2
        if len(point_in_block) > 0:
            point_in_block = point_in_block % 4
            for id in point_in_block:
                #nếu có người nào đó có tài nguyên để cướp thì sang phase 9, ko thì về phase 2
                if np.sum(env_state[int(ALL_INFOR_PLAYER*id) : int(ALL_INFOR_PLAYER*(id+1))][CARD_1_PLAYER_INDEX:CARD_1_PLAYER_INDEX+CARD_PLAYER_LEN]) > 0:
                    env_state[PHASE] = 9
                    break
        return env_state
    elif phase_env == 8:
        # print('tai nguyen: ', env_state[ALL_INFOR_PLAYER*id_action:ALL_INFOR_PLAYER*(id_action+1)][CARD_1_PLAYER_INDEX:CARD_1_PLAYER_INDEX+CARD_PLAYER_LEN], 'cần bỏ thêm: ', env_state[CARD_NEED_DROP])
        source_drop = action - 73
        #trừ tài nguyên người chơi và trả vào ngân hàng
        env_state[ALL_INFOR_PLAYER*id_action + CARD_1_PLAYER_INDEX + source_drop] -= 1
        env_state[CARD_BANK_INDEX + source_drop] += 1
        env_state[ALL_INFOR_PLAYER*id_action + 1] -= 1          #giảm tổng số lượng thẻ
        env_state[CARD_NEED_DROP] -= 1
        if env_state[CARD_NEED_DROP] == 0:
            #nếu đã trả đủ
            id_next = id_action
            for id in range(1, 5):
                #kiểm tra các người chơi kế tiếp, ai nhiều hơn 7 lá thì phải bỏ bài, ai ít hơn thì thôi, nếu quay lại đến người chơi chính thì đến phase đănt knight
                id_next = (id_action + id) % 4 
                if id_next == env_state[MAIN_PLAYER]:
                    env_state[PHASE] = 7            #update 12h39 ngày 28/8
                    env_state[ID_ACTION] = env_state[MAIN_PLAYER]
                    break
                else:
                    if np.sum(env_state[ALL_INFOR_PLAYER*id_next:ALL_INFOR_PLAYER*(id_next+1)][CARD_1_PLAYER_INDEX:CARD_1_PLAYER_INDEX+CARD_PLAYER_LEN]) > 7:
                        env_state[CARD_NEED_DROP] = np.sum(env_state[ALL_INFOR_PLAYER*id_next:ALL_INFOR_PLAYER*(id_next+1)][CARD_1_PLAYER_INDEX:CARD_1_PLAYER_INDEX+CARD_PLAYER_LEN]) // 2
                        env_state[ID_ACTION] = id_next
                        break
                    else:
                        pass
        
        return env_state 
    elif phase_env == 9:
        #trừ ngẫu nhiên 1 tài nguyên của người chơi bị cướp
        #các action tăng dần tương ứng với thứ tự người chơi 1,2,3 sau mình
        player_robbed = (action - 77 + id_action)%4
        # print('kêt quả cươp: ', action, env_state[MAIN_PLAYER], player_robbed, env_state[TURN])
        player_robbed_source = env_state[ALL_INFOR_PLAYER*player_robbed:ALL_INFOR_PLAYER*(player_robbed+1)][CARD_1_PLAYER_INDEX:CARD_1_PLAYER_INDEX+CARD_PLAYER_LEN]
        source_can_steal = np.where(player_robbed_source>0)[0]
        random_steal_source = np.random.choice(source_can_steal)        #xác nhận tài nguyên bị cướp
        #cập nhật tài nguyên người chơi
        env_state[ALL_INFOR_PLAYER*player_robbed + CARD_1_PLAYER_INDEX + random_steal_source] -= 1
        env_state[ALL_INFOR_PLAYER*player_robbed + 1] -= 1
        env_state[ALL_INFOR_PLAYER*id_action + CARD_1_PLAYER_INDEX + random_steal_source] += 1
        env_state[ALL_INFOR_PLAYER*id_action + 1] += 1
        #update tài nguyên xong về phase 2
        if env_state[CHECK_ROLL_INDEX] == 1:
            env_state[PHASE] = 1
        else:
            env_state[PHASE] = 2
        return env_state
    elif phase_env == 10:
        #nhận action, tăng tài nguyên lên, trừ ở ngân hàng đi, nếu còn được lấy thì chuyển phase 10, ko thì về phase2
        source_get = action - 86
        env_state[ALL_INFOR_PLAYER*id_action + CARD_1_PLAYER_INDEX + source_get] += 1
        env_state[ALL_INFOR_PLAYER*id_action + 1] += 1
        env_state[CARD_BANK_INDEX + source_get] -= 1
        if env_state[USE_YEAR_OF_PLENTY] != 0 and np.sum(env_state[CARD_BANK_INDEX : CARD_BANK_INDEX+CARD_BANK_LEN]) > 0:
            env_state[USE_YEAR_OF_PLENTY] = 0
        else:
            if env_state[CHECK_ROLL_INDEX] == 1:
                env_state[PHASE] = 1
            else:
                env_state[PHASE] = 2
        return env_state
    elif phase_env == 11:
        #nhận action, trừ hết tài nguyên tương ứng của các ng chơi khác rồi cộng cho ng chơi hiện tại, sau đó về phase 2
        source_get = action - 86
        sum_steal_monopoly = 0
        for id in range(1,4):
            id_robbed = (id+id_action)%4
            sum_steal_monopoly += env_state[int(ALL_INFOR_PLAYER*id_robbed + CARD_1_PLAYER_INDEX + source_get)]
            env_state[int(ALL_INFOR_PLAYER*id_robbed + CARD_1_PLAYER_INDEX + source_get)] = 0
        env_state[int(ALL_INFOR_PLAYER*id_action + CARD_1_PLAYER_INDEX + source_get)] += sum_steal_monopoly
        env_state[ATTRIBUTE_PLAYER_1_INDEX + 1] = np.sum(env_state[CARD_1_PLAYER_INDEX : CARD_1_PLAYER_INDEX + CARD_BANK_LEN])
        env_state[ATTRIBUTE_PLAYER_2_INDEX + 1] = np.sum(env_state[CARD_2_PLAYER_INDEX : CARD_2_PLAYER_INDEX + CARD_BANK_LEN])
        env_state[ATTRIBUTE_PLAYER_3_INDEX + 1] = np.sum(env_state[CARD_3_PLAYER_INDEX : CARD_3_PLAYER_INDEX + CARD_BANK_LEN])
        env_state[ATTRIBUTE_PLAYER_4_INDEX + 1] = np.sum(env_state[CARD_4_PLAYER_INDEX : CARD_4_PLAYER_INDEX + CARD_BANK_LEN])
        if env_state[CHECK_ROLL_INDEX] == 1:
            env_state[PHASE] = 1
        else:
            env_state[PHASE] = 2        
        return env_state
    elif phase_env == 12:
        #nhận action, cập nhật offer
        if action == 103:
            env_state[PHASE] = 13
        else:
            source_sell = action - 81
            index_in_list_offer = (id_action - env_state[MAIN_PLAYER])%4
            if id_action == env_state[MAIN_PLAYER]:
                env_state[int(OFFER_MAIN_INDEX + index_in_list_offer*OFFER_LEN + source_sell)] += 1
                # print(env_state[OFFER_MAIN_INDEX : OFFER_MAIN_INDEX + OFFER_LEN], env_state[int(ALL_INFOR_PLAYER*env_state[ID_ACTION] + 1)], np.sum(env_state[int(ALL_INFOR_PLAYER*env_state[ID_ACTION] + 5) :int(ALL_INFOR_PLAYER*env_state[ID_ACTION] + 10)]), '1111111111')
                if np.sum(env_state[OFFER_MAIN_INDEX : OFFER_MAIN_INDEX + OFFER_LEN]) == env_state[int(ALL_INFOR_PLAYER*env_state[ID_ACTION] + 1)]:        #hết tài nguyên rồi thì next phase
                    env_state[PHASE] = 13
            # else:
            #     env_state[int(OFFER_MAIN_INDEX + index_in_list_offer*OFFER_LEN + CARD_BANK_LEN + source_sell)] += 1
            #     if np.where(env_state[int(OFFER_MAIN_INDEX + index_in_list_offer*OFFER_LEN) : int(OFFER_MAIN_INDEX + (index_in_list_offer+1)*OFFER_LEN)]) == env_state[int(ALL_INFOR_PLAYER*env_state[ID_ACTION] + 1)]:        #hết tài nguyên rồi thì next phase
            #         env_state[PHASE] = 13
            

        return env_state
    elif phase_env == 13:
        #nhận action, cập nhật offer
        if action == 103:
            #DONE deal
            # if check_trade_with_harbor(env_state) and env_state[ID_ACTION] == env_state[MAIN_PLAYER]:
            #     #nếu deal đổi vs cảng hay ngân hàng đc, thực hiện vs ngân hàng r về phase 2
            #     main_deal = env_state[OFFER_MAIN_INDEX : OFFER_MAIN_INDEX + OFFER_LEN]
            #     env_state[int(ALL_INFOR_PLAYER*id_action + CARD_1_PLAYER_INDEX) : int(ALL_INFOR_PLAYER*id_action + CARD_1_PLAYER_INDEX+ CARD_BANK_LEN)] = env_state[int(ALL_INFOR_PLAYER*id_action + CARD_1_PLAYER_INDEX) : int(ALL_INFOR_PLAYER*id_action + CARD_1_PLAYER_INDEX+ CARD_BANK_LEN)] + main_deal[CARD_BANK_LEN:] - main_deal[:CARD_BANK_LEN]
            #     env_state[CARD_BANK_INDEX : CARD_BANK_INDEX + CARD_BANK_LEN] = env_state[CARD_BANK_INDEX : CARD_BANK_INDEX + CARD_BANK_LEN] - main_deal[CARD_BANK_LEN:] + main_deal[:CARD_BANK_LEN]
            #     env_state[OFFER_MAIN_INDEX : OFFER_MAIN_INDEX+OFFER_LEN] = np.zeros(OFFER_LEN)      #cập nhật lại thành deal trống
            #     env_state[int(ALL_INFOR_PLAYER*id_action+1)] = np.sum(env_state[int(ALL_INFOR_PLAYER*id_action + CARD_1_PLAYER_INDEX) : int(ALL_INFOR_PLAYER*id_action + CARD_1_PLAYER_INDEX+ CARD_BANK_LEN)])
            #     env_state[NUMBER_TRADE_OF_PLAYER] += 1
            #     env_state[PHASE] = 2
            # else:
            id_next = (id_action + 1)%4
            env_state[ID_ACTION] = id_next
            if id_next == env_state[MAIN_PLAYER]:
                env_state[PHASE] = 14
            else:
                env_state[PHASE] = 15
        else:
            #tiếp tục nạp vào deal
            source_buy = action - 86
            index_in_list_offer = (id_action - env_state[MAIN_PLAYER])%4
            if id_action == env_state[MAIN_PLAYER]: 
                env_state[int(OFFER_MAIN_INDEX + index_in_list_offer*OFFER_LEN + CARD_BANK_LEN + source_buy)] += 1
                all_number_res = env_state[1:POINT_INDEX:ALL_INFOR_PLAYER].copy()
                all_number_res[int(env_state[MAIN_PLAYER])] = 0
                if np.sum(env_state[OFFER_MAIN_INDEX+CARD_BANK_LEN : OFFER_MAIN_INDEX+OFFER_LEN]) >= np.max(all_number_res):
                    id_next = (id_action + 1)%4
                    env_state[ID_ACTION] = id_next
                    env_state[PHASE] = 15
            # else: 
            #     env_state[int(OFFER_MAIN_INDEX + index_in_list_offer*OFFER_LEN + source_buy)] += 1
            #     if np.sum(env_state[int(OFFER_MAIN_INDEX + index_in_list_offer*OFFER_LEN) : int(OFFER_MAIN_INDEX + index_in_list_offer*OFFER_LEN)+CARD_BANK_LEN]) > env_state[int(env_state[MAIN_PLAYER]*ALL_INFOR_PLAYER+1)]:
            #         #neu ng choi phu redeal
            #         id_next = (id_action + 1)%4
            #         env_state[ID_ACTION] = id_next
            #         if id_next == env_state[MAIN_PLAYER]:
            #             env_state[PHASE] = 14
            #         else:
            #             env_state[PHASE] = 15
        return env_state
    elif phase_env == 14:
        #cập nhật thông tin nếu có trao đổi sau đấy về phase 2
        if action == 104:
            #bỏ qua các deal khác, về phase 2
            env_state[OFFER_MAIN_INDEX:OFFER_MAIN_INDEX + OFFER_LEN*4] = np.zeros(OFFER_LEN*4)
            env_state[PHASE] = 2
        else:
            #nếu người chơi chính đồng ý trao đổi
            # if 90 < action and action < 94:
            id_trading = (id_action + action - 90)%4
            deal_traded = env_state[OFFER_MAIN_INDEX + OFFER_LEN*(action - 90):OFFER_MAIN_INDEX + OFFER_LEN*(action - 90+1)]
            #cập nhật tài nguyên của người được đồng ý trade
            env_state[ALL_INFOR_PLAYER*id_trading + CARD_1_PLAYER_INDEX : int(ALL_INFOR_PLAYER*id_trading + CARD_1_PLAYER_INDEX+ CARD_BANK_LEN)] = deal_traded[:CARD_BANK_LEN] + env_state[ALL_INFOR_PLAYER*id_trading + CARD_1_PLAYER_INDEX:ALL_INFOR_PLAYER*id_trading + CARD_1_PLAYER_INDEX+ CARD_BANK_LEN] - deal_traded[CARD_BANK_LEN:]
            #cập nhật tài nguyên của người chơi chính
            env_state[ALL_INFOR_PLAYER*id_action + CARD_1_PLAYER_INDEX : int(ALL_INFOR_PLAYER*id_action + CARD_1_PLAYER_INDEX+ CARD_BANK_LEN)] = deal_traded[CARD_BANK_LEN:] + env_state[ALL_INFOR_PLAYER*id_action + CARD_1_PLAYER_INDEX:ALL_INFOR_PLAYER*id_action + CARD_1_PLAYER_INDEX+ CARD_BANK_LEN] - deal_traded[:CARD_BANK_LEN]
            env_state[ALL_INFOR_PLAYER*id_trading + 1] = np.sum(env_state[ALL_INFOR_PLAYER*id_trading + CARD_1_PLAYER_INDEX : int(ALL_INFOR_PLAYER*id_trading + CARD_1_PLAYER_INDEX+ CARD_BANK_LEN)])
            env_state[ALL_INFOR_PLAYER*id_action + 1] = np.sum(env_state[ALL_INFOR_PLAYER*id_action + CARD_1_PLAYER_INDEX : int(ALL_INFOR_PLAYER*id_action + CARD_1_PLAYER_INDEX+ CARD_BANK_LEN)])
            #làm mới lại các deal
            env_state[OFFER_MAIN_INDEX:OFFER_MAIN_INDEX + OFFER_LEN*4] = np.zeros(OFFER_LEN*4)
            env_state[PHASE] = 2      
        return env_state
    elif phase_env == 15:
        index_in_list_offer = (id_action - env_state[MAIN_PLAYER])%4
        if action == 104:
            #nếu người chơi ko đồng ý, trả về offer rỗng tương ứng, nhảy sang người chơi khác
            env_state[int(OFFER_MAIN_INDEX + index_in_list_offer*OFFER_LEN) : int(OFFER_MAIN_INDEX + (1 + index_in_list_offer)*OFFER_LEN)] = np.zeros(OFFER_LEN)
            id_next = (id_action + 1)%4
            env_state[ID_ACTION] = id_next
            if id_next == env_state[MAIN_PLAYER]:
                env_state[PHASE] = 14
                
        elif action == 105:
            #nếu người chơi đồng ý, đặt offer của người chơi chính thành của người chơi này
            main_deal = env_state[OFFER_MAIN_INDEX : OFFER_MAIN_INDEX + OFFER_LEN]
            env_state[int(OFFER_MAIN_INDEX + index_in_list_offer*OFFER_LEN) : int(OFFER_MAIN_INDEX + (1 + index_in_list_offer)*OFFER_LEN)] = main_deal
            id_next = (id_action + 1)%4
            env_state[ID_ACTION] = id_next
            if id_next == env_state[MAIN_PLAYER]:
                env_state[PHASE] = 14
        # else:      #update tắt redeal 08/09/2022
        #     #nếu đi tạo deal mới, nạp tài nguyên vào deal rồi sang phase 12
        #     source_sell = action - 81
        #     env_state[int(OFFER_MAIN_INDEX + index_in_list_offer*OFFER_LEN + CARD_BANK_LEN + source_sell)] += 1
        #     env_state[PHASE] = 12
        return env_state
    elif phase_env == 16:
        #nhậc action, xác định soso lượn tài nguyên đem đổi
        source_sell = action - 81
        index_in_list_offer = (id_action - env_state[MAIN_PLAYER])%4
        house_data = env_state[POINT_INDEX: POINT_INDEX+POINT_LEN]
        point_in_port = house_data[POINT_IN_HARBOR]
        player_port = np.where((point_in_port == id_action) | (point_in_port == (id_action + 4)))[0]     #tìm các điểm là cảng của người chơi
        number_res_sell = 4
        player_source = env_state[int(ALL_INFOR_PLAYER*id_action+ATTRIBUTE_PLAYER) : int(ALL_INFOR_PLAYER*id_action+ATTRIBUTE_PLAYER+CARD_PLAYER_LEN)]
        type_of_port = env_state[TYPE_HARBOR_INDEX : TYPE_HARBOR_INDEX+TYPE_HARBOR_LEN]
        for id in player_port:
            type_port = type_of_port[id//2]
            if type_port == 0 and player_source[source_sell] > 2:
                number_res_sell = 3
            elif type_port != 0 and player_source[source_sell] > 1 and source_sell == (type_port-1):
                number_res_sell = 2
                break
        env_state[int(OFFER_MAIN_INDEX + index_in_list_offer*OFFER_LEN + source_sell)] += number_res_sell
        env_state[PHASE] = 17
        return env_state
    elif phase_env == 17:
        #nhận action, cập nhật tài nguyên cho ng chơi và ngân hàng
        source_get = action - 86
        main_deal = env_state[OFFER_MAIN_INDEX : OFFER_MAIN_INDEX + OFFER_LEN]
        # print(main_deal, 'checkkkkkkkk')
        source_sell = np.where(main_deal > 0)[0][0]
        env_state[ALL_INFOR_PLAYER*id_action + CARD_1_PLAYER_INDEX + source_get] += 1
        env_state[ALL_INFOR_PLAYER*id_action + CARD_1_PLAYER_INDEX + source_sell] -= main_deal[source_sell]
        env_state[ALL_INFOR_PLAYER*id_action + 1] += 1 - main_deal[source_sell]
        env_state[CARD_BANK_INDEX + source_get] -= 1       
        env_state[CARD_BANK_INDEX + source_sell] += main_deal[source_sell]   
        env_state[OFFER_MAIN_INDEX:OFFER_MAIN_INDEX + OFFER_LEN] = np.zeros(OFFER_LEN)          #làm mới lại deal
        env_state[PHASE] = 2
        return env_state



def one_game(list_player_, per_file):
    env_state = initEnv()
    temp_file = [[0] for i in range(getAgentSize())]
    count_turn = 0

    while count_turn < 15000:
        action, temp_file, per_file = action_player(env_state,list_player_,temp_file,per_file)     
        # print_mode_action(action, env_state)
        env_state = stepEnv(env_state, action)
        # print_mode_board(action, env_state)
        count_turn += 1
        if check_winner(env_state) != -1:
            score = env_state[ATTRIBUTE_PLAYER_1_INDEX:ATTRIBUTE_PLAYER_4_INDEX+2:ALL_INFOR_PLAYER].copy()
            vitory_card = env_state[CARD_EFFECT_1_PLAYER_INDEX+CARD_EFFECT_PLAYER_LEN-1:CARD_EFFECT_4_PLAYER_INDEX+CARD_EFFECT_PLAYER_LEN+1:ALL_INFOR_PLAYER].copy()
            env_state[ATTRIBUTE_PLAYER_1_INDEX:ATTRIBUTE_PLAYER_4_INDEX+2:ALL_INFOR_PLAYER] = score+vitory_card
            break
    
    winner = check_winner(env_state)
    # print('winner', winner)
    if winner == -1:
        pass
    else:
        for id_player in range(getAgentSize()):
            env_state[PHASE] = 1
            action, temp_file, per_file = action_player(env_state,list_player_,temp_file,per_file)
            # print(per_file)
            env_state[ID_ACTION] = (env_state[ID_ACTION] + 1)%4
    # score = env_state[ATTRIBUTE_PLAYER_1_INDEX:ATTRIBUTE_PLAYER_4_INDEX+2:ALL_INFOR_PLAYER]
    # vitory_card = env_state[CARD_EFFECT_1_PLAYER_INDEX+CARD_EFFECT_PLAYER_LEN-1:CARD_EFFECT_4_PLAYER_INDEX+CARD_EFFECT_PLAYER_LEN+1:ALL_INFOR_PLAYER]
    # score_real = score+vitory_card
    # print('điểm các người chơi: ', score_real)
    # print('Số lần truyền vào player:', count_turn)
    return winner, per_file


def normal_main(list_player, times, per_file):
    count = np.zeros(len(list_player)+1)
    all_id_player = np.arange(len(list_player))
    for van in range(times):
        shuffle = np.random.choice(all_id_player, getAgentSize(), replace=False)
        shuffle_player = [list_player[shuffle[0]], list_player[shuffle[1]], list_player[shuffle[2]], list_player[shuffle[3]]]
        winner, per_file = one_game(shuffle_player, per_file)
        if winner == -1:
            count[winner] += 1
        else:
            count[shuffle[winner]] += 1
    return list(count.astype(np.int64)), per_file

@njit()
def getReward(player_state):
    all_score = np.array([player_state[P_SCORE], player_state[P_P1_ATTRIBUTE_PLAYER], player_state[P_P2_ATTRIBUTE_PLAYER], player_state[P_P3_ATTRIBUTE_PLAYER]])
    # print(all_score)
    if np.max(all_score) < 10:
        return -1 
    else:
        if np.argmax(all_score) == 0:
            return 1
        else:
            return 0