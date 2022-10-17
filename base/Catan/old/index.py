import numpy as np
INDEX = 0

ATTRIBUTE_PLAYER = 5  #score, number_source_card, number_dev_card, number_knight_used, longest_road_of_player
CARD_PLAYER_LEN = 5
CARD_EFFECT_PLAYER_LEN = 5
INDEX_NUMBER_KNIGHT_USED_IN_ATTRIBUTE = 3
#Nguyên liệu để mua các loại thẻ
ROAD = np.array([1, 1, 0, 0, 0])
SETTLEMENT = np.array([1, 1, 1, 1, 0])
CITY = np.array([0, 0, 0, 2, 3]) 
DEV_CARD = np.array([0, 0, 1, 1, 1]) #dev card


# Người chơi 1
ATTRIBUTE_PLAYER_1_INDEX = INDEX
INDEX += ATTRIBUTE_PLAYER
CARD_1_PLAYER_INDEX = INDEX
INDEX += CARD_PLAYER_LEN
CARD_EFFECT_1_PLAYER_INDEX = INDEX
INDEX += CARD_EFFECT_PLAYER_LEN

# Người chơi 2
ATTRIBUTE_PLAYER_2_INDEX = INDEX
INDEX += ATTRIBUTE_PLAYER
CARD_2_PLAYER_INDEX = INDEX
INDEX += CARD_PLAYER_LEN
CARD_EFFECT_2_PLAYER_INDEX = INDEX
INDEX += CARD_EFFECT_PLAYER_LEN

# Người chơi 3
ATTRIBUTE_PLAYER_3_INDEX = INDEX
INDEX += ATTRIBUTE_PLAYER
CARD_3_PLAYER_INDEX = INDEX
INDEX += CARD_PLAYER_LEN
CARD_EFFECT_3_PLAYER_INDEX = INDEX
INDEX += CARD_EFFECT_PLAYER_LEN

# Người chơi 4
ATTRIBUTE_PLAYER_4_INDEX = INDEX
INDEX += ATTRIBUTE_PLAYER
CARD_4_PLAYER_INDEX = INDEX
INDEX += CARD_PLAYER_LEN
CARD_EFFECT_4_PLAYER_INDEX = INDEX
INDEX += CARD_EFFECT_PLAYER_LEN

# Độ dài thông tin của 1 người chơi 
ALL_INFOR_PLAYER = ATTRIBUTE_PLAYER_2_INDEX - ATTRIBUTE_PLAYER_1_INDEX
# Các điểm đặt nhà
POINT_INDEX = INDEX
POINT_LEN = 54
INDEX +=POINT_LEN

# Các con đường
ROAD_INDEX = INDEX
ROAD_LEN = 72
INDEX += ROAD_LEN

#vị trí của robber
ROBBER_BLOCK_INDEX = INDEX
ROBBER_BLOCK_LEN = 1
INDEX += ROBBER_BLOCK_LEN

# Số trên các ô đất
TOKEN_BLOCK_INDEX = INDEX
TOKEN_BLOCK_LEN = 19
INDEX += TOKEN_BLOCK_LEN

# Loại tài nguyên trên các ô đất
TYPE_SOURCE_INDEX = INDEX
TYPE_SOURCE_LEN = 19
INDEX += TYPE_SOURCE_LEN

# Loại cảng
TYPE_HARBOR_INDEX = INDEX
TYPE_HARBOR_LEN = 9
INDEX += TYPE_HARBOR_LEN

# Thẻ nguyên liệu
CARD_BANK_INDEX = INDEX
CARD_BANK_LEN = 5
INDEX += CARD_BANK_LEN

# Thẻ chức năng trong ngân hàng
CARD_EFFECT_BANK_INDEX = INDEX
CARD_EFFECT_BANK_LEN = 25
INDEX += CARD_EFFECT_BANK_LEN

# OFFER
OFFER_LEN = 10

OFFER_MAIN_INDEX = INDEX
INDEX += OFFER_LEN

OFFER_1_INDEX = INDEX
INDEX += OFFER_LEN

OFFER_2_INDEX = INDEX
INDEX += OFFER_LEN

OFFER_3_INDEX = INDEX
INDEX += OFFER_LEN

#OTHER INFORMATION

OTHER_IN4_INDEX = INDEX
TEMP_POINT = OTHER_IN4_INDEX
MAIN_PLAYER = OTHER_IN4_INDEX + 1 
LAST_ROLL = OTHER_IN4_INDEX + 2
PHASE = OTHER_IN4_INDEX + 3
ID_ACTION = OTHER_IN4_INDEX + 4

USE_BUILD_ROAD = OTHER_IN4_INDEX + 5
USE_YEAR_OF_PLENTY = OTHER_IN4_INDEX + 6
PLAYER_CAN_USE_DEV_CARD = OTHER_IN4_INDEX + 7
CARD_NEED_DROP = OTHER_IN4_INDEX + 8
LONGEST_ROAD_PLAYER = OTHER_IN4_INDEX + 9
LARGEST_ARMY_PLAYER = OTHER_IN4_INDEX + 10
NUMBER_TRADE_OF_PLAYER = OTHER_IN4_INDEX + 11
TURN = OTHER_IN4_INDEX + 12
# USED_DEV = OTHER_IN4_INDEX + 13
CHECK_ROLL_INDEX = OTHER_IN4_INDEX + 13

OTHER_IN4_LEN = 14
'''[temp_point, main_player, last_roll, phase, id_action, use_build_road, 
    use_year_of_plenty, card_need_drop, longest_road_player, largest_army, number_trade, turn]
'''
INDEX += OTHER_IN4_LEN


P1_SETTLEMENT_1ST = INDEX
P1_SETTLEMENT_2ST = INDEX + 1
P2_SETTLEMENT_1ST = INDEX + 2
P2_SETTLEMENT_2ST = INDEX + 3
P3_SETTLEMENT_1ST = INDEX + 4
P3_SETTLEMENT_2ST = INDEX + 5
P4_SETTLEMENT_1ST = INDEX + 6
P4_SETTLEMENT_2ST = INDEX + 7

OTHER_SETTLEMENT_PLAYER_INFOR_LEN = 8
INDEX += OTHER_SETTLEMENT_PLAYER_INFOR_LEN
END_INDEX_FIRST_SETTLEMENT = INDEX


'''
    Player_state index
    Index của player
'''
#INFOR PLAYER
P_ATTRIBUTE_PLAYER_LEN = 5

P_INDEX = 0
P_SCORE = P_INDEX
P_NUMBER_SOURCE_CARD = P_INDEX + 1
P_NUMBER_DEV_CARD = P_INDEX + 2
P_NUMBER_KNIGHT_USED = P_INDEX + 3
P_LONGEST_ROAD_OF_PLAYER = P_INDEX + 4

P_INDEX += P_ATTRIBUTE_PLAYER_LEN
P_CARD_PLAYER_INDEX = P_INDEX
P_INDEX += CARD_PLAYER_LEN

P_DEV_CARD = P_INDEX
P_INDEX += CARD_EFFECT_PLAYER_LEN

#OTHER_PLAYER
#score, number_source_card, number_dev_card, number_knight_used, longest_road_of_player
P_P1_ATTRIBUTE_PLAYER = P_INDEX
P_INDEX += P_ATTRIBUTE_PLAYER_LEN

P_P2_ATTRIBUTE_PLAYER = P_INDEX
P_INDEX += P_ATTRIBUTE_PLAYER_LEN

P_P3_ATTRIBUTE_PLAYER = P_INDEX
P_INDEX += P_ATTRIBUTE_PLAYER_LEN

#Các điểm đặt nhà,thành phố
P_POINT_INDEX = P_INDEX
P_INDEX += POINT_LEN

#Các điểm đặt đường
P_ROAD_INDEX = P_INDEX
P_INDEX += ROAD_LEN

#Vị trí của quân cướp
P_ROBBER_BLOCK_INDEX = P_INDEX
P_INDEX += ROBBER_BLOCK_LEN

# Số trên các ô đất
P_TOKEN_BLOCK_INDEX = P_INDEX
P_INDEX += TOKEN_BLOCK_LEN

# Nguyên liệu trên các ô đất
P_TYPE_SOURCE_INDEX = P_INDEX
P_INDEX += TYPE_SOURCE_LEN

# 9 cảng
P_TYPE_HARBOR_INDEX = P_INDEX
P_INDEX += TYPE_HARBOR_LEN

#kiểm tra ngân hàng còn tài nguyên không
P_SOURCE_IN_BANK_INDEX = P_INDEX
P_SOURCE_IN_BANK_LEN = 5
P_INDEX += P_SOURCE_IN_BANK_LEN

#kiểm tra bank còn thẻ dev ko
P_DEV_CARD_IN_BANK_INDEX = P_INDEX
P_DEV_CARD_SOURCE_IN_BANK_LEN = 1
P_INDEX += P_DEV_CARD_SOURCE_IN_BANK_LEN

#Offer của người chơi chính đưa ra
P_OFFER_MAIN_INDEX = P_INDEX
P_INDEX += OFFER_LEN

#Offer của người chơi 1, liền sau mình (nếu ko phải người chơi chính thì đây là offer của người chơi đưa ra)
P_OFFER_1_INDEX = P_INDEX
P_INDEX += OFFER_LEN

#Offer của người chơi 2
P_OFFER_2_INDEX = P_INDEX
P_INDEX += OFFER_LEN

#Offer của người chơi 3
P_OFFER_3_INDEX = P_INDEX
P_INDEX += OFFER_LEN

#Những thông tin khác
# [temp_point, main_player, last_roll, phase, id_action, use_build_road, use_year_of_plenty, turn]
P_TEMP_POINT = P_INDEX #Điểm thứ nhất khi chọn đường
P_MAIN_PLAYER = P_INDEX + 1 #main player
P_LAST_ROLL = P_INDEX + 2 #last_roll, giá trị xúc sắc gần nhất
P_PHASE = P_INDEX + 3 #Phase
P_ID_ACTION = P_INDEX + 4 #ID_action
P_USE_BUILD_ROAD = P_INDEX + 5 # Thông tin dùng thẻ chức năng có thêm 2 đường 
P_USE_YEAR_OF_PLENTY = P_INDEX + 6 #Thông tin dùng thẻ chức năng có thêm 2 nguyên liệu
P_PLAYER_CAN_USE_DEV_CARD = P_INDEX + 7
P_CARD_NEED_DROP = P_INDEX + 8
P_LONGEST_ROAD_PLAYER = OTHER_IN4_INDEX + 9
P_LARGEST_ARMY_PLAYER = OTHER_IN4_INDEX + 10
P_NUMBER_TRADE_OF_PLAYER = P_INDEX + 11
P_TURN = P_INDEX + 12
# P_USED_DEV = P_INDEX + 10
P_INDEX += 13

POINT_ROAD_RELATIVE = np.array([
    [0, 1, -1],
    [1, 2, 3],
    [3, 4, -1],
    [4, 5, -1],
    [5, 6, 7],
    [7, 8, -1],
    [8, 9, 10],
    [10, 11, -1],
    [11, 12, -1],
    [12, 13, 14],
    [14, 15, -1],
    [15, 16, 17],
    [17, 18, -1],
    [18, 19, -1],
    [19, 20, 21],
    [21, 22, -1],
    [22, 23, 24],
    [24, 25, -1],
    [25, 26, -1],
    [26, 27, 28],
    [28, 29, -1],
    [29, 30, 31],
    [31, 32, -1],
    [32, 33, -1],
    [33, 34, 35],
    [35, 36, -1],
    [36, 37, 38],
    [38, 39, -1],
    [39, 40, -1],
    [40, 41, 0],
    [41, 42, 65],
    [42, 37, 43],
    [43, 44, 45],
    [45, 46, 34],
    [46, 47, 30],
    [47, 48, 49],
    [49, 50, 27],
    [50, 51, 23],
    [51, 52, 53],
    [53, 54, 20],
    [54, 55, 16],
    [55, 56, 57],
    [57, 58, 13],
    [58, 59, 9],
    [59, 60, 61],
    [61, 62, 6],
    [63, 62, 2],
    [63, 64, 65],
    [64, 66, 71],
    [66, 67, 44],
    [67, 68, 48],
    [68, 69, 52],
    [69, 70, 56],
    [70, 71, 60]])

POINT_POINT_RELATIVE = np.array([
    [29, 1, -1], 
    [46, 2, 0], 
    [1, 3, -1], 
    [4, 2, -1], 
    [3, 45, 5], 
    [6, 4, -1], 
    [5, 43, 7], 
    [8, 6, -1], 
    [7, 9, -1], 
    [10, 8, 42], 
    [9, 11, -1], 
    [12, 10, 40], 
    [11, 13, -1], 
    [12, 14, -1], 
    [39, 15, 13], 
    [14, 16, -1], 
    [37, 17, 15], 
    [16, 18, -1], 
    [19, 17, -1], 
    [18, 36, 20], 
    [21, 19, -1], 
    [20, 34, 22], 
    [23, 21, -1], 
    [22, 24, -1], 
    [25, 23, 33], 
    [24, 26, -1], 
    [27, 25, 31], 
    [26, 28, -1], 
    [27, 29, -1], 
    [30, 0, 28], 
    [29, 31, 47], 
    [32, 30, 26], 
    [31, 33, 49], 
    [34, 32, 24], 
    [33, 21, 35], 
    [36, 50, 34], 
    [35, 19, 37], 
    [16, 38, 36], 
    [51, 37, 39], 
    [14, 40, 38], 
    [41, 39, 11], 
    [40, 42, 52], 
    [43, 41, 9], 
    [42, 6, 44], 
    [45, 53, 43], 
    [44, 4, 46], 
    [1, 47, 45], 
    [48, 46, 30], 
    [47, 49, 53], 
    [50, 48, 32], 
    [49, 35, 51], 
    [38, 52, 50], 
    [53, 51, 41], 
    [52, 44, 48]])

ROAD_ROAD_RELATIVE = np.array([
    [1, 40, 41, -1], 
    [0, 2, 3, -1], 
    [1, 3, 62, 63], 
    [1, 2, 4, -1], 
    [3, 5, -1, -1], 
    [4, 6, 7, -1], 
    [5, 7, 61, 62], 
    [5, 6, 8, -1], 
    [7, 9, 10, -1], 
    [8, 10, 58, 59], 
    [8, 9, 11, -1], 
    [10, 12, -1, -1], 
    [11, 13, 14, -1], 
    [12, 14, 57, 58], 
    [12, 13, 15, -1], 
    [14, 16, 17, -1],
    [15, 17, 54, 55], 
    [15, 16, 18, -1], 
    [17, 19, -1, -1], 
    [18, 20, 21, -1], 
    [19, 21, 53, 54], 
    [19, 20, 22, -1], 
    [21, 23, 24, -1], 
    [22, 24, 50, 51],
    [22, 23, 25, -1], 
    [24, 26, -1, -1], 
    [25, 27, 28, -1], 
    [26, 28, 49, 50], 
    [26, 27, 29, -1], 
    [28, 30, 31, -1], 
    [29, 31, 46, 47], 
    [29, 30, 32, -1], 
    [31, 33, -1, -1], 
    [32, 34, 35, -1], 
    [33, 35, 45, 46], 
    [33, 34, 36, -1],
    [35, 37, 38, -1], 
    [42, 43, 36, 38],
    [36, 37, 39, -1], 
    [38, 40, -1, -1], 
    [39, 0, 41, -1], 
    [40, 0, 65, 42], 
    [41, 65, 43, 37], 
    [37, 42, 44, 45], 
    [43, 66, 67, 45], 
    [43, 44, 46, 34], 
    [34, 45, 47, 30], 
    [46, 48, 49, 30], 
    [67, 68, 49, 47], 
    [47, 48, 50, 27], 
    [49, 51, 23, 27], 
    [52, 53, 23, 50],
    [68, 69, 53, 51],  
    [52, 54, 20, 51], 
    [55, 16, 20, 53], 
    [56, 57, 16, 54], 
    [70, 57, 55, 69], 
    [58, 13, 55, 56], 
    [59, 9, 13, 57], 
    [61, 9, 58, 60], 
    [61, 59, 70, 71], 
    [62, 6, 59, 60], 
    [2, 6, 61, 63], 
    [2, 62, 64, 65], 
    [65, 63, 71, 66], 
    [41, 63, 64, 42], 
    [64, 71, 67, 44], 
    [44, 66, 68, 48], 
    [67, 69, 52, 48], 
    [70, 56, 52, 68],
    [71, 60, 56, 69], 
    [64, 60, 70, 66]])

BLOCK_BLOCK_RELATIVE = np.array([
    [1, 17, 12, 11, -1, -1],
    [0, 2, 17, -1, -1, -1],
    [1, 3, 16, 17, -1, -1],
    [2, 4, 16, -1, -1, -1],
    [3, 5, 15, 16, -1, -1],
    [4, 6, 15, -1, -1, -1],
    [5, 7, 14, 15, -1, -1],
    [6, 8, 14, -1, -1, -1],
    [7, 9, 13, 14,-1, -1],
    [8, 10, 13, -1, -1, -1],
    [9, 11, 12, 13, -1, -1],
    [0, 10, 12, -1, -1, -1],
    [0, 10, 11, 13, 17, 18],
    [8, 9, 10, 12, 14, 18],
    [6, 7, 8, 13, 15, 18],
    [4, 5, 6, 14, 16, 18],
    [2, 3, 4, 15, 17, 18],
    [0, 1, 2, 12, 16, 18],
    [12, 13, 14, 15, 16, 17]])

ROAD_BY_POINT = np.array([
        [29, 0], #1
        [0, 1], 
        [1, 46],
        [1, 2],
        [2, 3],
        [3, 4],
        [45, 4],
        [4, 5],
        [5, 6],
        [43, 6],
        [6, 7],
        [7, 8],
        [8, 9],
        [42, 9],
        [9, 10],
        [10, 11],
        [40, 11],
        [11, 12],
        [12, 13],
        [13, 14],
        [39, 14],
        [14, 15],
        [15, 16],
        [37, 16],
        [16, 17],
        [17, 18],
        [18, 19],
        [36, 19],
        [19, 20],
        [20, 21],
        [34, 21],
        [21, 22],
        [22, 23],
        [23, 24],
        [33, 24],
        [24, 25],
        [25, 26],
        [31, 26],
        [26, 27],
        [27, 28],
        [28, 29],
        [29, 30],
        [30, 31],
        [31, 32],
        [49, 32],
        [32, 33],
        [33, 34],
        [34, 35],
        [50, 35],
        [35, 36],
        [36, 37],
        [37, 38],
        [51, 38],
        [38, 39],
        [39, 40],
        [40, 41],
        [52, 41],
        [41, 42],
        [42, 43],
        [43, 44],
        [53, 44],
        [44, 45],
        [45, 46],
        [46, 47],
        [47, 48],
        [30, 47],
        [48, 49],
        [49, 50],
        [50, 51],
        [51, 52],
        [52, 53],
        [53, 48]])

POINT_IN_BLOCK = np.array([[0, 1, 29, 30, 46, 47], [1, 2, 3, 4, 45, 46], [4, 5, 6, 43, 44, 45],
                [6, 7, 8, 9, 42, 43], [9, 10, 11, 40, 41, 42], [11, 12, 13, 14, 39, 40],
                [14, 15, 16, 37, 38, 39], [16, 17, 18, 19, 36, 37], [19, 20, 21, 34, 35, 36],
                [21, 22, 23, 24, 33, 34], [24, 25, 26, 31, 32, 33],[26, 27, 28, 29, 30, 31],
                [30, 31, 32, 47, 48, 49],[32, 33, 34, 35, 49, 50],[35, 36, 37, 38, 50, 51],
                [38, 39, 40, 41, 51, 52],[41, 42, 43, 44, 52, 53],[44, 45, 46, 47, 48, 53],[48, 49, 50, 51, 52, 53]])

POINT_IN_HARBOR_2D = np.array([[0, 1], [4, 5], [7, 8], [10, 11], [14, 15], [17, 18], [20, 21], [24, 25], [27, 28]])

POINT_IN_HARBOR = np.array([ 0,  1,  4,  5,  7,  8, 10, 11, 14, 15, 17, 18, 20, 21, 24, 25, 27, 28])
