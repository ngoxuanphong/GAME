import numpy as np
from numba import njit
from numba.typed import List

TILE_TILE = np.array(
    [[1, 11, 12, 17, -1, -1],  # 0
     [0, 2, 17, -1, -1, -1],  # 1
     [1, 3, 16, 17, -1, -1],  # 2
     [2, 4, 16, -1, -1, -1],  # 3
     [3, 5, 15, 16, -1, -1],  # 4
     [4, 6, 15, -1, -1, -1],  # 5
     [5, 7, 14, 15, -1, -1],  # 6
     [6, 8, 14, -1, -1, -1],  # 7
     [7, 9, 13, 14, -1, -1],  # 8
     [8, 10, 13, -1, -1, -1],  # 9
     [9, 11, 12, 13, -1, -1],  # 10
     [0, 10, 12, -1, -1, -1],  # 11
     [0, 10, 11, 13, 17, 18],  # 12
     [8, 9, 10, 12, 14, 18],  # 13
     [6, 7, 8, 13, 15, 18],  # 14
     [4, 5, 6, 14, 16, 18],  # 15
     [2, 3, 4, 15, 17, 18],  # 16
     [0, 1, 2, 12, 16, 18],  # 17
     [12, 13, 14, 15, 16, 17]]  # 18
)

POINT_POINT = np.array(
    [[1, 29, -1],  # 0
     [0, 2, 46],  # 1
     [1, 3, -1],  # 2
     [2, 4, -1],  # 3
     [3, 5, 45],  # 4
     [4, 6, -1],  # 5
     [5, 7, 43],  # 6
     [6, 8, -1],  # 7
     [7, 9, -1],  # 8
     [8, 10, 42],  # 9
     [9, 11, -1],  # 10
     [10, 12, 40],  # 11
     [11, 13, -1],  # 12
     [12, 14, -1],  # 13
     [13, 15, 39],  # 14
     [14, 16, -1],  # 15
     [15, 17, 37],  # 16
     [16, 18, -1],  # 17
     [17, 19, -1],  # 18
     [18, 20, 36],  # 19
     [19, 21, -1],  # 20
     [20, 22, 34],  # 21
     [21, 23, -1],  # 22
     [22, 24, -1],  # 23
     [23, 25, 33],  # 24
     [24, 26, -1],  # 25
     [25, 27, 31],  # 26
     [26, 28, -1],  # 27
     [27, 29, -1],  # 28
     [0, 28, 30],  # 29
     [29, 31, 47],  # 30
     [26, 30, 32],  # 31
     [31, 33, 49],  # 32
     [24, 32, 34],  # 33
     [21, 33, 35],  # 34
     [34, 36, 50],  # 35
     [19, 35, 37],  # 36
     [16, 36, 38],  # 37
     [37, 39, 51],  # 38
     [14, 38, 40],  # 39
     [11, 39, 41],  # 40
     [40, 42, 52],  # 41
     [9, 41, 43],  # 42
     [6, 42, 44],  # 43
     [43, 45, 53],  # 44
     [4, 44, 46],  # 45
     [1, 45, 47],  # 46
     [30, 46, 48],  # 47
     [47, 49, 53],  # 48
     [32, 48, 50],  # 49
     [35, 49, 51],  # 50
     [38, 50, 52],  # 51
     [41, 51, 53],  # 52
     [44, 48, 52]]  # 53
)

ROAD_POINT = np.array(
    [[0, 29],  # 0
     [0,  1],  # 1
     [1, 46],  # 2
     [1,  2],  # 3
     [2,  3],  # 4
     [3,  4],  # 5
     [4, 45],  # 6
     [4,  5],  # 7
     [5,  6],  # 8
     [6, 43],  # 9
     [6,  7],  # 10
     [7,  8],  # 11
     [8,  9],  # 12
     [9, 42],  # 13
     [9, 10],  # 14
     [10, 11],  # 15
     [11, 40],  # 16
     [11, 12],  # 17
     [12, 13],  # 18
     [13, 14],  # 19
     [14, 39],  # 20
     [14, 15],  # 21
     [15, 16],  # 22
     [16, 37],  # 23
     [16, 17],  # 24
     [17, 18],  # 25
     [18, 19],  # 26
     [19, 36],  # 27
     [19, 20],  # 28
     [20, 21],  # 29
     [21, 34],  # 30
     [21, 22],  # 31
     [22, 23],  # 32
     [23, 24],  # 33
     [24, 33],  # 34
     [24, 25],  # 35
     [25, 26],  # 36
     [26, 31],  # 37
     [26, 27],  # 38
     [27, 28],  # 39
     [28, 29],  # 40
     [29, 30],  # 41
     [30, 31],  # 42
     [31, 32],  # 43
     [32, 49],  # 44
     [32, 33],  # 45
     [33, 34],  # 46
     [34, 35],  # 47
     [35, 50],  # 48
     [35, 36],  # 49
     [36, 37],  # 50
     [37, 38],  # 51
     [38, 51],  # 52
     [38, 39],  # 53
     [39, 40],  # 54
     [40, 41],  # 55
     [41, 52],  # 56
     [41, 42],  # 57
     [42, 43],  # 58
     [43, 44],  # 59
     [44, 53],  # 60
     [44, 45],  # 61
     [45, 46],  # 62
     [46, 47],  # 63
     [47, 48],  # 64
     [30, 47],  # 65
     [48, 49],  # 66
     [49, 50],  # 67
     [50, 51],  # 68
     [51, 52],  # 69
     [52, 53],  # 70
     [48, 53]]  # 71
)

POINT_ROAD = np.array(
    [[0,  1, -1],  # 0
     [1,  2,  3],  # 1
     [3,  4, -1],  # 2
     [4,  5, -1],  # 3
     [5,  6,  7],  # 4
     [7,  8, -1],  # 5
     [8,  9, 10],  # 6
     [10, 11, -1],  # 7
     [11, 12, -1],  # 8
     [12, 13, 14],  # 9
     [14, 15, -1],  # 10
     [15, 16, 17],  # 11
     [17, 18, -1],  # 12
     [18, 19, -1],  # 13
     [19, 20, 21],  # 14
     [21, 22, -1],  # 15
     [22, 23, 24],  # 16
     [24, 25, -1],  # 17
     [25, 26, -1],  # 18
     [26, 27, 28],  # 19
     [28, 29, -1],  # 20
     [29, 30, 31],  # 21
     [31, 32, -1],  # 22
     [32, 33, -1],  # 23
     [33, 34, 35],  # 24
     [35, 36, -1],  # 25
     [36, 37, 38],  # 26
     [38, 39, -1],  # 27
     [39, 40, -1],  # 28
     [0, 40, 41],  # 29
     [41, 42, 65],  # 30
     [37, 42, 43],  # 31
     [43, 44, 45],  # 32
     [34, 45, 46],  # 33
     [30, 46, 47],  # 34
     [47, 48, 49],  # 35
     [27, 49, 50],  # 36
     [23, 50, 51],  # 37
     [51, 52, 53],  # 38
     [20, 53, 54],  # 39
     [16, 54, 55],  # 40
     [55, 56, 57],  # 41
     [13, 57, 58],  # 42
     [9, 58, 59],  # 43
     [59, 60, 61],  # 44
     [6, 61, 62],  # 45
     [2, 62, 63],  # 46
     [63, 64, 65],  # 47
     [64, 66, 71],  # 48
     [44, 66, 67],  # 49
     [48, 67, 68],  # 50
     [52, 68, 69],  # 51
     [56, 69, 70],  # 52
     [60, 70, 71]]  # 53
)

PORT_POINT = np.array(
    [[0, 1],
     [4, 5],
     [7, 8],
     [10, 11],
     [14, 15],
     [17, 18],
     [20, 21],
     [24, 25],
     [27, 28]]
)

POINT_TILE = np.array(
    [[0, -1, -1],  # 0
     [0, 1, -1],  # 1
     [1, -1, -1],  # 2
     [1, -1, -1],  # 3
     [1, 2, -1],  # 4
     [2, -1, -1],  # 5
     [2, 3, -1],  # 6
     [3, -1, -1],  # 7
     [3, -1, -1],  # 8
     [3, 4, -1],  # 9
     [4, -1, -1],  # 10
     [4, 5, -1],  # 11
     [5, -1, -1],  # 12
     [5, -1, -1],  # 13
     [5, 6, -1],  # 14
     [6, -1, -1],  # 15
     [6, 7, -1],  # 16
     [7, -1, -1],  # 17
     [7, -1, -1],  # 18
     [7, 8, -1],  # 19
     [8, -1, -1],  # 20
     [8, 9, -1],  # 21
     [9, -1, -1],  # 22
     [9, -1, -1],  # 23
     [9, 10, -1],  # 24
     [10, -1, -1],  # 25
     [10, 11, -1],  # 26
     [11, -1, -1],  # 27
     [11, -1, -1],  # 28
     [0, 11, -1],  # 29
     [0, 11, 12],  # 30
     [10, 11, 12],  # 31
     [10, 12, 13],  # 32
     [9, 10, 13],  # 33
     [8, 9, 13],  # 34
     [8, 13, 14],  # 35
     [7, 8, 14],  # 36
     [6, 7, 14],  # 37
     [6, 14, 15],  # 38
     [5, 6, 15],  # 39
     [4, 5, 15],  # 40
     [4, 15, 16],  # 41
     [3, 4, 16],  # 42
     [2, 3, 16],  # 43
     [2, 16, 17],  # 44
     [1, 2, 17],  # 45
     [0, 1, 17],  # 46
     [0, 12, 17],  # 47
     [12, 17, 18],  # 48
     [12, 13, 18],  # 49
     [13, 14, 18],  # 50
     [14, 15, 18],  # 51
     [15, 16, 18],  # 52
     [16, 17, 18]]  # 53
)

TILE_POINT = np.array(
    [[0,  1, 29, 30, 46, 47],  # 0
     [1,  2,  3,  4, 45, 46],  # 1
     [4,  5,  6, 43, 44, 45],  # 2
     [6,  7,  8,  9, 42, 43],  # 3
     [9, 10, 11, 40, 41, 42],  # 4
     [11, 12, 13, 14, 39, 40],  # 5
     [14, 15, 16, 37, 38, 39],  # 6
     [16, 17, 18, 19, 36, 37],  # 7
     [19, 20, 21, 34, 35, 36],  # 8
     [21, 22, 23, 24, 33, 34],  # 9
     [24, 25, 26, 31, 32, 33],  # 10
     [26, 27, 28, 29, 30, 31],  # 11
     [30, 31, 32, 47, 48, 49],  # 12
     [32, 33, 34, 35, 49, 50],  # 13
     [35, 36, 37, 38, 50, 51],  # 14
     [38, 39, 40, 41, 51, 52],  # 15
     [41, 42, 43, 44, 52, 53],  # 16
     [44, 45, 46, 47, 48, 53],  # 17
     [48, 49, 50, 51, 52, 53]]  # 18
)

ROAD_PRICE = np.array([1, 1, 0, 0, 0])
SETTLEMENT_PRICE = np.array([1, 1, 1, 1, 0])
CITY_PRICE = np.array([0, 0, 0, 2, 3])
DEV_PRICE = np.array([0, 0, 1, 1, 1])

LEN_P_STATE = 210
AMOUNT_ACTION = 106

# Càng cao thì chạy càng lâu, nhưng tỉ lệ không end được game càng thấp
MAX_TURN_IN_ONE_GAME = 1200


@njit()
def initEnv():
    env = np.zeros(255)

    # [0:19]: Tài nguyên trên các ô đất
    temp = np.array([5, 0, 0, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 1, 1, 1, 4, 4, 4])
    np.random.shuffle(temp)
    env[0:19] = temp

    # [19]: Ví trí Robber
    env[19] = np.argmax(temp)

    # [20:39]: Số trên các ô đất
    temp_prob = np.zeros(19)
    temp = np.ones(19)
    temp[int(env[19])] = 0
    temp_1 = temp.copy()
    for i in range(4):
        temp_2 = np.where(temp == 1)[0]
        k = temp_2[np.random.randint(0, len(temp_2))]
        temp[k] = 0
        temp_1[k] = 0
        if i < 2:
            temp_prob[k] = 6
        else:
            temp_prob[k] = 8

        for j in TILE_TILE[k]:
            if j == -1:
                break
            else:
                temp[j] = 0

    temp = np.array([3, 3, 4, 4, 5, 5, 9, 9, 10, 10, 11, 11, 2, 12])
    np.random.shuffle(temp)
    temp_prob[np.where(temp_1 == 1)[0]] = temp
    env[20:39] = temp_prob

    # [39:48]: Các cảng
    temp = np.array([5, 5, 5, 5, 0, 1, 2, 3, 4])
    np.random.shuffle(temp)
    env[39:48] = temp

    # [48:53]: Tài nguyên ngân hàng
    env[48:53] = 19

    # [53:58]: Thẻ dev bank
    env[53:58] = np.array([14, 2, 2, 2, 5])

    # Thông tin người chơi: [58,100], [100:142], [142,184], [184:226]
    for p_idx in range(4):
        s_ = 58 + 42*p_idx  # 58, 100, 142, 184
        # [+0:+5]: Tài nguyên
        # [+5:+10]: Thẻ dev
        # [+10]: Điểm

        # [+11:+26]: Đường
        # [+26:+31]: Nhà
        # [+31:+35]: Thành phố
        env[s_+11:s_+35] = -1

        # [+35]: Số thẻ knight đã dùng
        # [+36]: Con đường dài nhất

        # [+37:+42]: Tỉ lệ trao đổi với Bank
        env[s_+37:s_+42] = 4

    # [226]: Danh hiệu quân đội mạnh nhất
    # [227]: Danh hiệu con đường dài nhất
    # [228]: Tổng xx
    env[226:229] = -1

    # [229]: Pha
    # [230]: Turn

    # [231]: Điểm đặt thứ nhất
    env[231] = -1

    # [232]: Số tài nguyên trả do bị chia

    # [233]: Đang dùng thẻ dev gì
    env[233] = -1

    # [234]: Số lần sử dụng thẻ dev
    # [235:239]: Loại thẻ dev được sử dụng trong turn hiện tại
    # [239]: Số lần tạo trade offer
    # [240:245]: Tài nguyên đưa ra trong trade offer
    # [245:250]: Tài nguyên yêu cầu trong trade offer
    # 250: Lượng tài nguyên được yêu cầu tối đa trong trade offer

    # [251:254]: Phản hồi của người chơi phụ về trade offer (đồng ý hay không)
    env[251:254] = -1

    # [254]: Người chơi đang action (không hẳn là người chơi chính)

    return env


@njit
def getAgentState(env: np.ndarray):
    p_state = np.zeros(LEN_P_STATE)

    p_idx = int(env[254])
    turn = env[230]
    if turn >= 4 and turn <= 7:
        main_p_idx = 7 - turn
    else:
        main_p_idx = turn % 4

    # [0:48]: Tài nguyên trên các ô đất, Vị trí Robber, Số trên các ô đất, Các cảng
    p_state[0:48] = env[0:48]

    # [48:53]: Tài nguyên Bank dạng 0: không, 1: có
    p_state[48:53] = env[48:53] > 0

    # [53]: Thẻ dev Bank dạng không hoặc có
    p_state[53] = (env[53:58] > 0).any()

    # [54:96]: Thông tin cá nhân
    # ##########
    # [+0:+5]: Tài nguyên
    # [+5:+10]: Thẻ dev
    # [+10]: Điểm
    # [+11:+26]: Đường
    # [+26:+31]: Nhà
    # [+31:+35]: Thành phố
    # [+35]: Số thẻ knight đã dùng
    # [+36]: Con đường dài nhất
    # [+37:+42]: Tỉ lệ trao đổi với Bank
    # ##########
    s_ = 58 + 42*p_idx
    p_state[54:96] = env[s_:s_+42]

    # Thông tin người chơi khác: [96:125], [125:154], [154:183]
    for i in range(1, 4):
        e_idx = (p_idx + i) % 4
        s_e = 58 + 42*e_idx
        s_p = 96 + 29*(i-1)  # 96, 125, 154

        # [+0]: Tổng tài nguyên
        p_state[s_p] = np.sum(env[s_e:s_e+5])

        # [+1]: Tổng số thẻ dev
        p_state[s_p+1] = np.sum(env[s_e+5:s_e+10])

        # [+2]: Điểm
        # [+3:+18]: Đường
        # [+18:+23]: Nhà
        # [+23:+27]: Thành phố
        # +27: Số thẻ knight đã dùng
        # +28: Con đường dài nhất
        p_state[s_p+2:s_p+29] = env[s_e+10:s_e+37]

    # [183]: Danh hiệu quân đội mạnh nhất
    # [184]: Danh hiệu con đường dài nhất
    for i in range(2):
        temp = env[226+i]
        if temp == -1:
            p_state[183+i] = -1
        else:
            p_state[183+i] = (temp - p_idx) % 4

    # [185]: Tổng xx
    # [186]: Pha
    p_state[185:187] = env[228:230]

    # [187]: Điểm đặt thứ nhất
    # [188]: Số tài nguyên phải bỏ do bị chia
    # [189]: Đang dùng thẻ dev gì
    # [190]: Số lần dùng thẻ dev
    # [191:195]: Loại thẻ dev được sử dụng trong turn hiện tại
    # [195]: Số lần trạo trade offer
    p_state[187:196] = env[231:240]

    # [196:201]: Tài nguyên đưa ra trong trade offer
    # [201:206]: Tài nguyên yêu cầu trong trade offer
    # [206:209]: Phản hồi của người chơi phụ
    # [209]: Người chơi chính
    p_state[209] = (main_p_idx - p_idx) % 4

    if p_state[209] != 0:  # Không phải người chơi chính
        p_state[196:201] = env[245:250]
        p_state[201:206] = env[240:245]
        p_state[206:209] = -1
    else:  # Người chơi chính
        p_state[196:206] = env[240:250]
        p_state[206:209] = env[251:254]

    return p_state


@njit
def get_p_point_n_all_road(p_state: np.ndarray):
    p_point = np.zeros(54)
    all_road = np.zeros(72)

    temp = p_state[65:80].astype(np.int32)
    for road in temp:
        if road == -1:
            break
        else:
            all_road[road] = 1
            p_point[ROAD_POINT[road]] = 1

    for j in range(3):
        s_ = 96 + 29*j

        temp = p_state[s_+3:s_+18].astype(np.int32)
        for road in temp:
            if road == -1:
                break
            else:
                all_road[road] = 1

        temp = p_state[s_+18:s_+23].astype(np.int32)
        for i in temp:
            if i == -1:
                break
            else:
                p_point[i] = 0

        temp = p_state[s_+23:s_+27].astype(np.int32)
        for i in temp:
            if i == -1:
                break
            else:
                p_point[i] = 0

    return p_point, all_road


@njit
def check_firstPoint(p_state: np.ndarray):
    p_point, all_road = get_p_point_n_all_road(p_state)
    list_point = np.where(p_point == 1)[0]
    for point in list_point:
        for road in POINT_ROAD[point]:
            if road != -1 and all_road[road] == 0:
                return True

    return False


@njit
def check_useDev(p_state: np.ndarray, list_action: np.ndarray):
    # Knight: Có là được dùng
    if p_state[191] == 1:
        list_action[55] = 1

    # Roadbuilding: Còn đường và còn vị trí xây đường
    if p_state[192] == 1 and p_state[79] == -1 and check_firstPoint(p_state):
        list_action[56] = 1

    # Yearofplenty: Ngân hàng có tài nguyên
    if p_state[193] == 1 and (p_state[48:53] == 1).any():
        list_action[57] = 1

    # Monopoly: Có là được dùng
    if p_state[194] == 1:
        list_action[58] = 1


@njit
def get_p_point_n_all_stm_city(p_state: np.ndarray):
    p_point = np.zeros(54)
    all_stm_and_city = np.zeros(54)

    temp = p_state[65:80].astype(np.int32)
    for i in temp:
        if i == -1:
            break
        else:
            p_point[ROAD_POINT[i]] = 1

    temp = p_state[80:85].astype(np.int32)
    for i in temp:
        if i == -1:
            break
        else:
            all_stm_and_city[i] = 1
            p_point[i] = 0

    temp = p_state[85:89].astype(np.int32)
    for i in temp:
        if i == -1:
            break
        else:
            all_stm_and_city[i] = 1
            p_point[i] = 0

    for j in range(3):
        s_ = 96 + 29*j

        temp = p_state[s_+18:s_+23].astype(np.int32)
        for i in temp:
            if i == -1:
                break
            else:
                all_stm_and_city[i] = 1
                p_point[i] = 0

        temp = p_state[s_+23:s_+27].astype(np.int32)
        for i in temp:
            if i == -1:
                break
            else:
                all_stm_and_city[i] = 1
                p_point[i] = 0

    return p_point, all_stm_and_city


@njit
def getValidActions(p_state: np.ndarray):
    phase = p_state[186]
    list_action = np.full(AMOUNT_ACTION, 0)

    if phase == 11:  # Yêu cầu tài nguyên khi trade với người
        # Nếu đã có ít nhất 1 tài nguyên, thì phải có action dừng
        if (p_state[201:206] > 0).any():
            list_action[104] = 1

        # Các action thêm tài nguyên: các loại tài nguyên mà không có trong phần đưa ra
        # list_action[59:64] = p_state[196:201] == 0
        for i in range(5):
            if p_state[196+i] == 0:
                list_action[59+i] = 1

        return list_action

    if phase == 6:  # Chọn các mô đun giữa turn
        check_useDev(p_state, list_action)

        if (p_state[54:59] >= ROAD_PRICE).all():
            # Mua đường (86)
            if p_state[79] == -1 and check_firstPoint(p_state):
                list_action[86] = 1

            # Mua nhà (87)
            if (p_state[54:59] >= SETTLEMENT_PRICE).all() and p_state[84] == -1:
                p_point, all_stm_and_city = get_p_point_n_all_stm_city(p_state)
                list_point = np.where(p_point == 1)[0]
                for point in list_point:
                    # list_road = POINT_ROAD[point][POINT_ROAD[point] != -1]
                    # nearest_points = ROAD_POINT[list_road].flatten()
                    # if (all_stm_and_city[nearest_points] == 0).all():
                    #     list_action[87] = 1
                    #     break
                    check = True
                    for road in POINT_ROAD[point]:
                        if road != -1:
                            for neares_point in ROAD_POINT[road]:
                                if all_stm_and_city[neares_point] == 1:
                                    check = False
                                    break

                            if not check:
                                break

                    if check:
                        list_action[87] = 1
                        break

        # Mua thành phố (88)
        if (p_state[54:59] >= CITY_PRICE).all() and p_state[80] != -1 and p_state[88] == -1:
            list_action[88] = 1

        # Mua thẻ dev (89)
        if (p_state[54:59] >= DEV_PRICE).all() and p_state[53] == 1:
            list_action[89] = 1

        if (p_state[54:59] > 0).any():
            # Trade với người (90)
            if p_state[195] > 0 and (p_state[np.array([96, 125, 154])] > 0).any():
                list_action[90] = 1

            # Trade với bank (91)
            if (p_state[54:59] >= p_state[91:96]).any():
                temp = np.where(p_state[54:59] >= p_state[91:96])[0]
                for res in temp:
                    for res_1 in range(5):
                        if res_1 != res and p_state[48+res_1] == 1:
                            list_action[91] = 1
                            break

                    if list_action[91] == 1:
                        break

        # Kết thúc lượt (92)
        list_action[92] = 1

        return list_action

    if phase == 10:  # Đưa ra tài nguyên khi trade với người
        # Nếu đã có ít nhất 1 tài nguyên, thì phải có action dừng
        if (p_state[196:201] > 0).any():
            list_action[103] = 1

        # Các action thêm tài nguyên: các tài nguyên mà bản thân có
        # list_action[95:100] = p_state[54:59] > p_state[196:201]
        for i in range(5):
            if p_state[54+i] > p_state[196+i]:
                list_action[95+i] = 1

        # Nếu số loại tài nguyên bỏ vào là 4 thì không cho bỏ loại thứ 5 vào
        if np.count_nonzero(p_state[196:201] > 0) == 4:
            list_action[95+np.argmin(p_state[196:201])] = 0

        return list_action

    if phase == 3:  # Trả tài nguyên do bị chia bài
        # list_action[95:100] = p_state[54:59] > 0
        for i in range(5):
            if p_state[54+i] > 0:
                list_action[95+i] = 1

        return list_action

    if phase == 12:  # Người chơi phụ phản hồi trade
        # Action từ chối: 93, Action: đồng ý: 94
        list_action[93:95] = 1

        # Vào pha này thì chắc chắn người chơi phụ phải có thể trade
        # if (p_state[54:59] >= p_state[196:201]).all():
        #     list_action[94] = 1

        return list_action

    if phase == 15:  # Chọn tài nguyên muốn nhận từ ngân hàng
        # Chọn những tài nguyên mà ngân hàng có, khác tài nguyên đưa ra
        list_action[59:64] = p_state[48:53]
        list_action[59+np.argmax(p_state[196:201])] = 0

        return list_action

    if phase == 14:  # Chọn tài nguyên khi trade với ngân hàng
        # Chọn những tài nguyên mà khi chọn, ngân hàng còn ít nhất 1 loại tài nguyên khác
        # temp = np.where(p_state[54:59] >= p_state[91:96])[0]
        # for res in temp:
        for res in range(5):
            if p_state[54+res] >= p_state[91+res]:
                for res_1 in range(5):
                    if res_1 != res and p_state[48+res_1] == 1:
                        list_action[95+res] = 1
                        break

        return list_action

    if phase == 1:  # Chọn các điểm đầu mút của đường
        if p_state[187] == -1:  # Chọn điểm thứ nhất
            p_point, all_road = get_p_point_n_all_road(p_state)
            list_point = np.where(p_point == 1)[0]
            for point in list_point:
                for road in POINT_ROAD[point]:
                    if road != -1 and all_road[road] == 0:
                        list_action[point] = 1
                        break

            return list_action

        all_road = np.zeros(72)

        temp = p_state[65:80].astype(np.int32)
        for i in temp:
            if i == -1:
                break
            else:
                all_road[i] = 1

        for j in range(3):
            s_ = 96 + 29*j

            temp = p_state[s_+3:s_+18].astype(np.int32)
            for i in temp:
                if i == -1:
                    break
                else:
                    all_road[i] = 1

        first_point = int(p_state[187])
        for road in POINT_ROAD[first_point]:
            if road != -1 and all_road[road] == 0:
                list_action[ROAD_POINT[road]] = 1

        list_action[first_point] = 0

        return list_action

    if phase == 4:  # Di chuyển Robber
        list_action[64:83] = 1
        list_action[int(64+p_state[19])] = 0

        return list_action

    if phase == 13:  # Người chơi chính duyệt trade
        # Action bỏ qua
        list_action[105] = 1

        # Chọn người để trade
        # Vào pha này thì chắc chắn có ít nhất một người đồng ý trade
        list_action[100:103] = p_state[206:209]

        return list_action

    if phase == 5:  # Chọn người để cướp tài nguyên
        robber_pos = int(p_state[19])
        for i in range(3):
            s_ = 96 + 29*i
            if p_state[s_] > 0:  # Chỉ xét khi có tài nguyên
                temp = p_state[s_+18:s_+27].astype(np.int32)
                for point in temp:
                    if point != -1 and point in TILE_POINT[robber_pos]:
                        list_action[83+i] = 1
                        break

        return list_action

    if phase == 2:  # Đổ xx hoặc dùng thẻ dev
        # Đổ xx
        list_action[54] = 1

        check_useDev(p_state, list_action)

        return list_action

    if phase == 0:  # Chọn điểm đặt nhà đầu game
        list_action[0:54] = 1

        temp = p_state[np.array(
            [80, 81, 114, 115, 143, 144, 172, 173])].astype(np.int32)
        for stm in temp:
            if stm != -1:
                list_action[stm] = 0
                for point in POINT_POINT[stm]:
                    if point != -1:
                        list_action[point] = 0

        return list_action

    if phase == 8:  # Chọn các điểm mua nhà
        p_point, all_stm_and_city = get_p_point_n_all_stm_city(p_state)
        list_point = np.where(p_point == 1)[0]
        for point in list_point:
            list_road = POINT_ROAD[point][POINT_ROAD[point] != -1]
            nearest_points = ROAD_POINT[list_road].flatten()
            if (all_stm_and_city[nearest_points] == 0).all():
                list_action[point] = 1

        return list_action

    if phase == 7:  # Chọn tài nguyên khi dùng thẻ dev
        if p_state[189] == 2:  # Đang dùng yearofplenty
            list_action[59:64] = p_state[48:53]
        elif p_state[189] == 3:  # Đang dùng monopoly
            list_action[59:64] = 1

        return list_action

    if phase == 9:  # Chọn các điểm mua thành phố
        temp = p_state[80:85].astype(np.int32)
        for p_stm in temp:
            if p_stm == -1:
                break
            else:
                list_action[p_stm] = 1

        return list_action

    return list_action


@njit
def find_max_road(len_: int, diemXP: int, duongDaDi: np.ndarray, p_road: np.ndarray, p_point: np.ndarray):
    duongCanCheck = np.zeros(72)
    for road in POINT_ROAD[diemXP]:
        if road != -1 and p_road[road] == 1 and duongDaDi[road] == 0:
            duongCanCheck[road] = 1

    list_duongCanCheck = np.where(duongCanCheck == 1)[0]
    if len(list_duongCanCheck) == 0 or p_point[diemXP] == 0:
        if len_ > 0:
            return len_

    max_len = len_
    for road in list_duongCanCheck:
        duongDaDi_copy = duongDaDi.copy()
        duongDaDi_copy[road] = 1
        diemXP_1 = 9999
        for diem in ROAD_POINT[road]:
            if diem != diemXP:
                diemXP_1 = diem
                break

        len_1 = find_max_road(
            len_+1, diemXP_1, duongDaDi_copy, p_road, p_point)
        if len_1 > max_len:
            max_len = len_1

    return max_len


@njit
def get_p_longest_road(env: np.ndarray, p_idx: int):
    s_ = 58 + 42*p_idx
    p_road = np.zeros(72)
    p_full_point = np.zeros(54)

    temp = env[s_+11:s_+26]
    list_road = temp[temp != -1].astype(np.int32)
    p_road[list_road] = 1
    p_full_point[ROAD_POINT[list_road].flatten()] = 1

    p_point = p_full_point.copy()
    for i in range(1, 4):
        e_idx = (p_idx + i) % 4
        s_e = 58 + 42*e_idx
        temp = env[s_e+26:s_e+35]
        stm_and_city = temp[temp != -1].astype(np.int32)
        p_point[stm_and_city] = 0

    duongDaDi = np.zeros(72)
    max_len = 0
    list_full_point = np.where(p_full_point == 1)[0]
    for point in list_full_point:
        len_ = find_max_road(0, point, duongDaDi, p_road, p_point)
        if len_ > max_len:
            max_len = len_

    return max_len


@njit
def roll_xx(env: np.ndarray):
    p_idx = int(env[254])
    dice1 = np.random.randint(1, 7)
    dice2 = np.random.randint(1, 7)
    env[228] = dice1 + dice2  # Cập nhật xx vào env

    if env[228] == 7:  # Check xem ai phải bỏ bài
        # Giả sử không ai bị chia bài, thì sẽ sang phase di chuyển Robber
        env[229] = 4
        for i in range(0, 4):
            e_idx = (p_idx + i) % 4
            s_e = 58 + 42*e_idx
            if np.sum(env[s_e:s_e+5]) > 7:  # Thừa, sang pha bỏ bài
                env[229] = 3  # Sang pha bỏ tài nguyên do bị chia
                # Cập nhật số lượng tài nguyên phải bỏ
                env[232] = np.sum(env[s_e:s_e+5]) // 2
                env[254] = e_idx  # Đổi người chơi action
                break

    else:  # Trả tài nguyên từ ngân hàng
        temp = np.where(env[20:39] == env[228])[0]
        list_tile = temp[temp != env[19]]
        p_res_receive = np.zeros((4, 5))
        for i in range(4):
            s_i = 58 + 42*i

            temp = env[s_i+26:s_i+31]
            p_stm = temp[temp != -1].astype(np.int32)

            temp = env[s_i+31:s_i+35]
            p_city = temp[temp != -1].astype(np.int32)

            for tile in list_tile:
                env_tile = int(env[tile])
                for stm in p_stm:
                    if stm in TILE_POINT[tile]:
                        p_res_receive[i][env_tile] += 1

                for city in p_city:
                    if city in TILE_POINT[tile]:
                        p_res_receive[i][env_tile] += 2

        total_res_receive = np.sum(p_res_receive, axis=0)

        # Chỉ trả những loại tài nguyên mà ngân hàng có đủ
        temp = np.where(env[48:53] < total_res_receive)[0]
        p_res_receive[:, temp] = 0
        total_res_receive = np.sum(p_res_receive, axis=0)

        # Trả tài nguyên
        env[48:53] -= total_res_receive  # Trừ tài nguyên ngân hàng
        for i in range(4):
            s_i = 58 + 42*i
            # Cộng tài nguyên cho người chơi
            env[s_i:s_i+5] += p_res_receive[i]

        # Sang phase chọn mô đun
        env[229] = 6


@njit
def update_new_stm(env: np.ndarray, new_stm_pos: int, s_: int):
    # Cập nhật tập nhà
    r_ = np.count_nonzero(env[s_+26:s_+31] != -1)
    env[s_+26+r_] = new_stm_pos

    # Cộng 1 điểm
    env[s_+10] += 1

    # Kiểm tra cảng
    for port in range(9):
        if new_stm_pos in PORT_POINT[port]:
            if env[39+port] == 5:
                env[s_+37:s_+42] = np.minimum(3, env[s_+37:s_+42])
            else:
                env[int(s_+37+env[39+port])] = 2

            break


@njit
def useDev(env: np.ndarray, action: int, s_: int, p_idx: int):
    env[s_+action-50] -= 1  # Giảm số lá dev của người chơi
    env[235:239] = 0  # Mỗi turn chỉ được sử dụng 1 thẻ dev
    if action == 55:  # Thẻ knight
        env[233] = 0
        env[234] = 1
        env[229] = 4  # Chuyển sang pha di chuyển Robber
        env[s_+35] += 1

        list_p_usedKnight = env[np.array([93, 135, 177, 219])]
        largestArmy = np.max(list_p_usedKnight)

        # Check lại danh hiệu
        if env[s_+35] >= 3 and env[s_+35] == largestArmy and env[226] != p_idx:
            # Nếu cddn < 3 thì ko cần xét do ko thể giành danh hiệu
            # Nếu cddn != longestRoad thì cũng ko cần xét do có xét cũng ko được danh hiệu
            # Nếu đang có danh hiệu thì việc xây đường giúp củng cố danh hiệu, nên ko cần xét
            list_p_check = np.where(
                list_p_usedKnight == largestArmy)[0]
            if len(list_p_check) == 1:  # Ghi nhận danh hiệu hoặc thay đổi danh hiệu
                if env[226] == -1:  # Chưa có danh hiệu
                    env[226] = p_idx  # Ghi nhận danh hiệu
                    env[s_+10] += 2  # Cộng điểm
                else:  # Thay đổi danh hiệu
                    # Trừ điểm người cũ
                    env[int(58+42*env[226]+10)] -= 2
                    env[226] = p_idx  # Ghi nhận người mới
                    env[s_+10] += 2  # Cộng điểm người cũ
            # Nếu có 2 người cùng có con đường dài nhất, không thay đổi danh hiệu

    elif action == 56:  # Roadbuilding
        env[233] = 1
        env[234] = 2
        env[229] = 1  # Chuyển về pha đặt đường

    elif action == 57:  # Yearofplenty
        env[233] = 2
        env[234] = 2
        env[229] = 7

    elif action == 58:  # Monopoly
        env[233] = 3
        env[234] = 1
        env[229] = 7


@njit
def after_useDev(env: np.ndarray):
    env[233] = -1
    env[234] = 0
    if env[228] == -1:  # Chưa roll xx
        env[229] = 2
        roll_xx(env)
    else:  # Đã roll xx
        env[229] = 6


@njit
def after_Rob(env: np.ndarray):
    if env[233] == 0:  # Đang dùng thẻ knight
        after_useDev(env)
    else:  # Vừa đổ ra 7
        env[229] = 6


@njit
def weighted_random(p: np.ndarray):
    a = np.sum(p)
    b = np.random.uniform(0, a)
    for i in range(len(p)):
        b -= p[i]
        if b <= 0:
            return i


@njit
def stepEnv(env: np.ndarray, action: int):
    phase = env[229]
    p_idx = int(env[254])
    s_ = 58 + 42*p_idx

    if phase == 11:  # Yêu cầu tài nguyên khi trade với người
        check = False
        if action == 104:  # Kết thúc
            check = True
        else:  # Thêm tài nguyên
            env[186+action] += 1
            if np.sum(env[245:250]) == env[250]:  # Lượng tài nguyên đạt đến tối đa
                check = True

        if check:
            env[250] = 0
            for i in range(1, 5):
                e_idx = (p_idx + i) % 4
                s_e = 58 + 42*e_idx

                if i == 4:  # Quay về người chơi chính, không có ai trade,
                    env[229] = 6  # Về luôn phase chọn mô đun
                    env[240:250] = 0  # Reset trade
                    env[251:254] = -1  # Reset phản hồi người chơi phụ
                    break

                if (env[s_e:s_e+5] < env[245:250]).any():
                    env[250+i] = 0
                else:
                    env[229] = 12
                    env[254] = e_idx
                    break

        return

    if phase == 6:  # Chọn các mô đun giữa turn
        if action == 92:  # Kết thúc lượt
            env[239] = 1  # Reset lại số lần tạo trade offer
            env[230] += 1  # Tăng turn
            n_idx = env[230] % 4
            s_n = int(58 + 42*n_idx)
            env[254] = n_idx  # Người đi tiếp theo

            # Kiểm tra thẻ dev có thể sử dụng
            env[235:239] = env[s_n+5:s_n+9] > 0
            # print(env[235:239], 'a6w5d6a5e65a6w')

            # Tổng xx
            env[228] = -1

            # Pha
            env[229] = 2

            if (env[235:239] == 0).all():
                roll_xx(env)

            return

        if action == 90:  # Trade với người
            env[239] -= 1  # Giảm số lần được tạo trade offer với người chơi
            env[229] = 10  # Sang pha đưa ra tài nguyên khi tạo trade

            # Kiểm tra lượng tài nguyên tối đa
            max_res = 0
            for i in range(1, 4):
                e_idx = (p_idx + i) % 4
                s_e = 58 + 42*e_idx
                a = np.sum(env[s_e:s_e+5])
                if a > max_res:
                    max_res = a

            env[250] = max_res  # Lượng tài nguyên tối đa được đưa vào trong trade

            return

        if action == 91:  # Trade với bank
            env[229] = 14

            return

        if action == 86:  # Mua đường
            env[s_:s_+5] -= ROAD_PRICE  # Trả tài nguyên
            env[48:53] += ROAD_PRICE
            env[229] = 1  # Chuyển sang pha đặt đường

            return

        if action == 89:  # Mua thẻ dev, không chuyển pha
            env[s_:s_+5] -= DEV_PRICE  # Trả tài nguyên mua thẻ dev
            env[48:53] += DEV_PRICE
            temp = env[53:58]

            dev_idx = weighted_random(temp)
            env[53+dev_idx] -= 1  # type: ignore # Trừ thẻ dev ở ngân hàng
            env[s_+5+dev_idx] += 1  # type: ignore # Cộng thẻ dev cho người chơi

            # Nếu thẻ dev vừa nhận là thẻ vp (dev_idx = 4) thì cộng điểm cho người chơi
            # if dev_idx == 4:
            #     env[s_+10] += 1
            # Note: Không được cộng do như thế các người khác sẽ biết

            return

        if action >= 55 and action < 59:  # Dùng thẻ dev
            useDev(env, action, s_, p_idx)

            return

        if action == 87:  # Mua nhà
            env[s_:s_+5] -= SETTLEMENT_PRICE
            env[48:53] += SETTLEMENT_PRICE
            env[229] = 8  # Sang pha mua nhà

            return

        if action == 88:  # Mua thành phố
            env[s_:s_+5] -= CITY_PRICE
            env[48:53] += CITY_PRICE
            env[229] = 9  # Chuyển sang pha mua thành phố

            return

    if phase == 10:  # Đưa ra tài nguyên khi trade với người
        if action == 103:  # Kết thúc
            env[229] = 11  # Chuyển sang pha yêu cầu tài nguyên

        else:  # Thêm tài nguyên
            env[145+action] += 1
            if np.sum(env[240:245]) == np.sum(env[s_:s_+5]):  # Đã bỏ hết tài nguyên
                env[229] = 11

        return

    if phase == 3:  # Trả tài nguyên do bị chia bài
        env[s_+action-95] -= 1  # Trừ tài nguyên của người chơi
        env[action-47] += 1  # Trả tài nguyên này cho ngân hàng

        env[232] -= 1  # Giảm số lượng thẻ cần trả
        if env[232] == 0:  # Đã trả đủ
            for i in range(1, 5):
                e_idx = (p_idx + i) % 4
                s_e = 58 + 42*e_idx
                if e_idx == env[230] % 4:  # Quay về người chơi chính
                    env[229] = 4  # Sang pha di chuyển Robber
                    env[254] = e_idx  # Chuyển người action
                    break
                else:
                    if np.sum(env[s_e:s_e+5]) > 7:
                        # Cập nhật số lá phải bỏ
                        env[232] = np.sum(env[s_e:s_e+5]) // 2
                        env[254] = e_idx  # Chuyển người action
                        break

        return

    if phase == 12:  # Người chơi phụ phản hồi trade
        r_ = int(p_idx - env[230]) % 4
        env[250+r_] = action - 93

        for i in range(1, 4):
            next_idx = (p_idx + i) % 4
            s_n = 58 + 42*next_idx
            env[254] = next_idx
            if next_idx == env[230] % 4:  # Chuyển về người chơi chính
                if (env[251:254] == 0).all():  # Không ai trade
                    env[229] = 6  # Về luôn phase chọn mô đun
                    env[240:250] = 0  # Reset trade
                    env[251:254] = -1  # Reset phản hồi người chơi phụ
                else:
                    env[229] = 13

                break

            if (env[s_n:s_n+5] < env[245:250]).any():
                env[250+r_+i] = 0
            else:
                env[229] = 12
                break

        # if r_ == 3:  # Chuyển về người chơi chính
        #     if (env[251:254] == 0).all():  # Không ai trade
        #         env[229] = 6  # Về luôn phase chọn mô đun
        #         env[240:250] = 0  # Reset trade
        #         env[251:254] = -1  # Reset phản hồi người chơi phụ
        #     else:
        #         env[229] = 13
        # Note: Đoạn này xử lí thiếu

        return

    if phase == 15:  # Chọn tài nguyên muốn nhận từ ngân hàng
        res_idx = action - 59

        # Người chơi
        env[s_+res_idx] += 1
        env[s_:s_+5] -= env[240:245]

        # Ngân hàng
        env[48+res_idx] -= 1
        env[48:53] += env[240:245]

        env[240:245] = 0
        env[229] = 6

        return

    if phase == 14:  # Chọn tài nguyên khi trade với ngân hàng
        res_idx = action - 95
        env[240+res_idx] = env[s_+37+res_idx]

        env[229] = 15

        return

    if phase == 1:  # Chọn các điểm đầu mút của đường
        if env[231] == -1:  # Chưa có điểm đặt thứ nhất
            env[231] = action

            return

        # Tìm con đường tương ứng và thay đổi tập đường
        for road in np.where(ROAD_POINT == env[231])[0]:
            if action in ROAD_POINT[road]:
                i = np.count_nonzero(env[s_+11:s_+26] != -1)
                env[s_+11+i] = road
                break

        # Check con đường dài nhất
        env[s_+36] = get_p_longest_road(env, p_idx)
        list_p_longestRoad = env[np.array([94, 136, 178, 220])]
        longestRoad = np.max(list_p_longestRoad)

        # Check lại danh hiệu
        if env[s_+36] >= 5 and env[s_+36] == longestRoad and env[227] != p_idx:
            # Nếu cddn < 5 thì ko cần xét do ko thể giành danh hiệu
            # Nếu cddn != longestRoad thì cũng ko cần xét do có xét cũng ko được danh hiệu
            # Nếu đang có danh hiệu thì việc xây đường giúp củng cố danh hiệu, nên ko cần xét
            list_p_check = np.where(list_p_longestRoad == longestRoad)[0]
            if len(list_p_check) == 1:  # Ghi nhận danh hiệu hoặc thay đổi danh hiệu
                if env[227] == -1:  # Chưa có danh hiệu
                    env[227] = p_idx  # Ghi nhận danh hiệu
                    env[s_+10] += 2  # Cộng điểm
                else:  # Thay đổi danh hiệu
                    env[int(58+42*env[227]+10)] -= 2  # Trừ điểm người cũ
                    env[227] = p_idx  # Ghi nhận người mới
                    env[s_+10] += 2  # Cộng điểm người cũ
            # Nếu có 2 người cùng có con đường dài nhất, không thay đổi danh hiệu

        # Xóa điểm đặt thứ nhất
        env[231] = -1

        # 8 turn đầu: Kết thúc turn, chuyển người chơi, nếu là turn 8 thì chỉ chuyển pha
        if env[230] <= 7:
            env[230] += 1
            if env[230] == 8:  # Chỉ chuyển pha
                env[239] = 1  # Cài lại số lần tạo trade offer
                env[229] = 2  # Sang pha 2: Đổ xx hoặc dùng thẻ dev
                roll_xx(env)  # Chưa thể có thẻ dev trong trường hợp này
            else:  # Thay đổi người chơi, chuyển pha
                env[229] = 0  # Chuyển sang pha đặt nhà đầu game
                if env[230] < 4:
                    env[254] = env[230]
                else:
                    env[254] = 7 - env[230]

        else:  # Các turn giữa game
            if env[223] != 1:  # Không dùng thẻ roadbuilding, về pha 6
                env[229] = 6

                return

            if env[233] == 1:  # Đang dùng thẻ roadbuilding
                env[234] -= 1
                # Hoặc là hết lượt dùng, hoặc đủ 15 đường, hoặc hết vị trí xây đường
                check = True
                # if env[234] == 0 or (env[s_+11:s_+26] != -1).all():
                if env[234] == 0 or env[s_+25] != -1:
                    check = False

                if check:
                    check = False
                    p_point = np.zeros(54)
                    all_road = np.zeros(72)

                    # Đường và điểm của người chơi
                    temp = env[s_+11:s_+26].astype(np.int32)
                    # list_road = temp[temp != -1].astype(np.int32)
                    # all_road[list_road] = 1
                    # p_point[ROAD_POINT[list_road].flatten()] = 1

                    for road in temp:
                        if road == -1:
                            break
                        else:
                            all_road[road] = 1
                            p_point[ROAD_POINT[road]] = 1

                    for i in range(1, 4):
                        e_idx = (p_idx + i) % 4
                        s_e = 58 + 42*e_idx

                        temp = env[s_e+11:s_e+26].astype(np.int32)
                        # list_road = temp[temp != -1].astype(np.int32)
                        # all_road[list_road] = 1
                        for road in temp:
                            if road == -1:
                                break
                            else:
                                all_road[road] = 1

                        temp = env[s_e+26:s_e+31].astype(np.int32)
                        # stm_and_city = temp[temp != -1].astype(np.int32)
                        # p_point[stm_and_city] = 0

                        for stm in temp:
                            if stm == -1:
                                break
                            else:
                                p_point[stm] = 0

                        temp = env[s_e+31:s_e+35].astype(np.int32)
                        for city in temp:
                            if city == -1:
                                break
                            else:
                                p_point[city] = 0

                    list_point = np.where(p_point == 1)[0]
                    for point in list_point:
                        # nearest_roads = POINT_ROAD[point][POINT_ROAD[point] != -1]
                        # if (all_road[nearest_roads] == 0).any():
                        #     check = True
                        #     break
                        for road in POINT_ROAD[point]:
                            if road != -1 and all_road[road] == 0:
                                check = True
                                break

                        if check:
                            break

                if not check:  # Kết thúc trạng thái sử dụng thẻ roadbuilding
                    after_useDev(env)

                return

        return

    if phase == 4:  # Di chuyển Robber
        robber_pos = action - 64
        env[19] = robber_pos  # Cập nhật vị trí Robber

        # Xét xem có cướp được tài nguyên của ai hay ko
        check = False
        for i in range(4):
            if i != p_idx:
                s_i = 58 + 42*i
                if (env[s_i:s_i+5] > 0).any():  # Nếu người này có tài nguyên
                    temp = env[s_i+26:s_i+35].astype(np.int32)
                    # stm_and_city = temp[temp != -1]
                    for point in temp:
                        if point != -1 and point in TILE_POINT[robber_pos]:
                            env[229] = 5
                            check = True
                            break

            if check:
                break

        if not check:  # Không cướp được của ai
            after_Rob(env)

        return

    if phase == 13:  # Người chơi chính duyệt trade
        if action == 105:  # Bỏ qua hết
            env[229] = 6  # Về luôn phase chọn mô đun
            env[240:250] = 0  # Reset trade
            env[251:254] = -1  # Reset phản hồi người chơi phụ

            return

        e_idx = (p_idx + action - 99) % 4
        s_e = 58 + 42*e_idx

        # Người chơi chính
        env[s_:s_+5] -= env[240:245]
        env[s_:s_+5] += env[245:250]

        # Người chơi phụ
        env[s_e:s_e+5] -= env[245:250]
        env[s_e:s_e+5] += env[240:245]

        env[229] = 6  # Về pha chọn mô đun
        env[240:250] = 0
        env[251:254] = -1  # Reset phản hồi người chơi phụ

        return

    if phase == 5:  # Chọn người để cướp tài nguyên
        e_idx = (p_idx + action - 82) % 4
        s_e = 58 + 42*e_idx
        temp = env[s_e:s_e+5]  # Tài nguyên

        res_idx = weighted_random(temp)
        env[s_e+res_idx] -= 1  # type: ignore # Trừ tài nguyên của người bị cướp
        env[s_+res_idx] += 1  # type: ignore # Cộng tài nguyên cho người cướp

        after_Rob(env)

        return

    if phase == 2:  # Đổ xx hoặc dùng thẻ dev
        if action == 54:  # Đổ xúc xắc
            roll_xx(env)

            return

        useDev(env, action, s_, p_idx)

        return

    if phase == 0:  # Chọn điểm đặt nhà đầu game
        update_new_stm(env, action, s_)

        # Nếu là lần 2 thì cộng tài nguyên
        if env[s_+10] == 2:
            temp = POINT_TILE[action][POINT_TILE[action] != -1]
            nearest_tiles = temp[env[temp] != 5]
            for tile in nearest_tiles:
                env_tile = int(env[tile])
                env[s_+env_tile] += 1  # Cộng tài nguyên cho người chơi
                env[48+env_tile] -= 1  # Trừ tài nguyên ở ngân hàng

        # Chuyển sang pha đặt đường
        env[231] = action  # Điểm đặt thứ nhất
        env[229] = 1  # Chuyển sang pha 1: Đặt đường

        return

    if phase == 8:  # Chọn các điểm mua nhà
        update_new_stm(env, action, s_)

        # Check xem nhà vừa xây có nằm trên con đường của ai khác hay ko
        check = False
        w_idx = 9999

        for i in range(1, 4):
            e_idx = (p_idx + i) % 4
            s_e = 58 + 42*e_idx

            temp = env[s_e+11:s_e+26]
            list_road = temp[temp != -1].astype(np.int32)
            count = 0
            for road in POINT_ROAD[action]:
                if road != -1 and road in list_road:
                    count += 1

            if count == 2:
                check = True
                w_idx = e_idx
                break

        if check:  # Người này vừa xây nhà cắt đường người khác
            s_w = 58 + 42*w_idx
            env[s_w+36] = get_p_longest_road(env, w_idx)
            list_p_longestRoad = env[np.array([94, 136, 178, 220])]
            longestRoad = np.max(list_p_longestRoad)

            if longestRoad < 5:
                if env[227] != -1:  # Có ông có danh hiệu, chứng tỏ vừa bị cướp
                    env[int(58+42*env[227]+10)] -= 2  # Trừ điểm
                    env[227] = -1  # Thu hồi danh hiệu
                #  Nếu chưa ai có danh hiệu thì bỏ qua
            else:
                list_p_check = np.where(list_p_longestRoad == longestRoad)[0]
                if len(list_p_check) == 1:  # Có đúng 1 ông có con đường dài nhất
                    if env[227] == -1:  # Chưa có ai có danh hiệu
                        env[227] = list_p_check[0]  # Bàn chơi ghi nhận
                        env[int(58+42*env[227]+10)] += 2  # Cộng điểm
                    else:  # Có ông đã có danh hiệu
                        if env[227] != list_p_check[0]:  # Thay đổi danh hiệu
                            env[int(58+42*env[227]+10)] -= 2  # Trừ điểm ông cũ
                            env[227] = list_p_check[0]  # Bàn chơi ghi nhận
                            # Cộng điểm ông mới
                            env[int(58+42*env[227]+10)] += 2
                        #  Nếu cùng là một người thì không thay đổi danh hiệu
                else:  # Có từ 2 ông trở lên cùng có con đường dài nhất
                    # Đang có người có danh hiệu, nhưng con đường không dài nhất
                    if env[227] != -1 and env[227] not in list_p_check:
                        env[int(58+42*env[227]+10)] -= 2  # Trừ điểm
                        env[227] = -1
                    # Hoặc là chưa ai có danh hiệu, hoặc là danh hiệu nằm trong list thì ko cần check

        env[229] = 6  # Về pha chọn mô đun

        return

    if phase == 7:  # Chọn tài nguyên khi dùng thẻ dev
        if env[233] == 2:  # Đang dùng thẻ yearofplenty
            env[action-11] -= 1  # Trừ tài nguyên cho ngân hàng
            env[s_+action-59] += 1  # Cộng tài nguyên cho người chơi

            # Giảm số lần sử dụng thẻ xuống
            env[234] -= 1

            # Nếu hết số lần sử dụng hoặc ngân hàng hết tài nguyên
            if env[234] == 0 or (env[48:53] == 0).all():
                after_useDev(env)

            return

        if env[233] == 3:  # Đang dùng thẻ monopoly
            sum_res = 0
            res_idx = action - 59
            for i in range(1, 4):
                e_idx = (p_idx + i) % 4
                s_e = 58 + 42*e_idx
                sum_res += env[s_e+res_idx]
                env[s_e+res_idx] = 0

            env[s_+res_idx] += sum_res
            after_useDev(env)

            return

    if phase == 9:  # Chọn các điểm mua thành phố
        # Cập nhật tập nhà
        r_ = np.count_nonzero(env[s_+31:s_+35] != -1)
        env[s_+31+r_] = action

        # Xóa nhà vừa được nâng cấp lên thành phố
        temp = env[s_+26:s_+31]
        for i in range(5):
            if temp[i] == action:
                temp[i] = -1
                break

        temp = temp[temp != -1]
        env[s_+26:s_+31] = -1
        env[s_+26:s_+26+len(temp)] = temp

        # Cộng điểm
        env[s_+10] += 1

        env[229] = 6  # Sang pha chọn mô đun

        return

    return


@njit
def checkEnded(env: np.ndarray):
    # Người chơi chỉ có thể chiến thắng khi tới lượt của mình
    # Kể cả trường hợp: A có 9 điểm, B có con đường dài nhất,
    # C có 9 điểm. C xây nhà phá đường của B, A được con đường
    # dài nhất, có 11 điểm, C chỉ có 10 điểm nhưng vẫn chiến thắng
    # mặc cho A có nhiều điểm hơn C
    ##
    main_p_idx = int(env[230]) % 4
    all_total_score = env[np.array([68, 110, 152, 194])] \
        + env[np.array([67, 109, 151, 193])]

    if all_total_score[main_p_idx] > 9:
        list_more_than_9 = np.where(all_total_score > 9)[0]
        max_score = np.max(all_total_score)

        # Chỉ có thể có tối đa 2 người chơi trên 9 điểm cùng lúc
        # Nếu có 2 người chơi trên 9 điểm thì không thể có người chơi có 12 điểm
        if len(list_more_than_9) == 2:
            # 2 người chơi có cùng số điểm cao nhất
            if (all_total_score[list_more_than_9] == max_score).all():
                return main_p_idx + 100

            # Người chơi chiến thắng có điểm cao nhất
            if all_total_score[main_p_idx] == max_score:
                return main_p_idx + 200

            # Người chơi chiến thắng không có điểm cao nhất
            return main_p_idx + 300

        # Đến đây thì tức là người chơi chính có điểm cao nhất
        if all_total_score[main_p_idx] == 12:
            return main_p_idx + 400

        # Chỉ đơn giản là chiến thắng với 10 hoặc 11 điểm, không có gì đặc biệt
        return main_p_idx

    return -1


@njit
def getReward(p_state: np.ndarray):
    main_p_idx = int(p_state[209])
    if main_p_idx == 0:  # Đến lượt của bản thân
        # Thắng hoặc chưa kết thúc game
        if p_state[64] > 9:  # Hơn 9 điểm => auto thắng
            return 1

        return -1  # Chưa có trên 9 điểm, chưa kết thúc game

    # Không phải lượt của bản thân
    # Thua hoặc chưa kết thúc game
    # Người chơi chính có hơn 9 điểm => thua
    if p_state[69 + 29*main_p_idx] > 9:
        return 0

    # Người chơi chính chưa có trên 9 điểm, chưa kết thúc game
    return -1


@njit
def getStateSize():
    return LEN_P_STATE


@njit
def getActionSize():
    return AMOUNT_ACTION


@njit
def getAgentSize():
    return 4


def one_game(list_player, per_file):
    env = initEnv()
    temp_file = [[0], [0], [0], [0]]

    winner = -1
    while env[230] < MAX_TURN_IN_ONE_GAME:
        p_idx = int(env[254])
        p_state = getAgentState(env)
        action, temp_file[p_idx], per_file = list_player[p_idx](
            p_state, temp_file[p_idx], per_file)
        stepEnv(env, action)
        winner = checkEnded(env)
        if winner != -1:
            break

    env[np.array([68, 110, 152, 194])] += env[np.array([67, 109, 151, 193])]

    if winner != -1:
        for i in range(4):
            env[254] = i
            env[229] = 2
            p_state = getAgentState(env)
            action, temp_file[i], per_file = list_player[i](
                p_state, temp_file[i], per_file)

    return winner, per_file




def normal_main(list_player, times, per_file):
    count_win = np.full(9, 0)
    # Mô tả count_win
    # 0, 1, 2, 3: Số trận thắng của p0, p1, p2, p3
    # 4: Số trận ko ai chiến thắng
    # 5: Số trận có 2 người cùng có điểm cao nhất\
    # 6: Số trận có 2 người trên 9 điểm, người chiến thắng có điểm cao nhất
    # 7: Số trận có 2 người trên 9 điểm, người chiến thắng không có điểm cao nhất
    # 8: Số trận mà người chiến thắng có 12 điểm

    p_lst_idx = np.arange(4)

    for _n in range(times):
        np.random.shuffle(p_lst_idx)
        winner, per_file = one_game(
            [list_player[p_lst_idx[0]], list_player[p_lst_idx[1]],
                list_player[p_lst_idx[2]], list_player[p_lst_idx[3]]], per_file
        )

        if winner == -1:
            count_win[4] += 1
        else:
            winner_idx = winner % 4
            count_win[p_lst_idx[winner_idx]] += 1
            special_case_idx = winner // 100
            if special_case_idx != 0:
                count_win[4+special_case_idx] += 1

    return list(count_win.astype(np.int64)), per_file


from system.mainFunc import dict_game_for_player, load_data_per2
game_name_ = 'Catan_v2'
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
    layer = np.zeros(getActionSize())
    for id in range(len(file_per_2[0])):
        layer += data_to_layer_NhatAnh_130922(state,file_per_2[0][id], file_per_2[1][id])
    base = np.zeros(getActionSize())
    actions = getValidActions(state)
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
    actions = getValidActions(state)
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
    return np.random.choice(np.where(getValidActions(state) == 1)[0])

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
    a = getValidActions(play_state)
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
    a = getValidActions(play_state)
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
    a = getValidActions(play_state)
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
def test2_An_130922(p_state,  file_per_2):
    list_action = getValidActions(p_state)
    list_action = np.where(list_action == 1)[0]
    action = neural_network_an_130922(p_state, file_per_2, list_action)
    return action

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
    list_action = getValidActions(p_state)
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
    list_action = getValidActions(p_state)
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
    list_action = getValidActions(state)
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
    norm_action = np.zeros(getActionSize())
    norm_action[list_action] = 1
    norm_action = norm_action.reshape(1, getActionSize())
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
    list_action = getValidActions(state)
    list_action = np.where(list_action == 1)[0]
    action = neural_network_hieu_130922(state, file_per_2[0], file_per_2[1], file_per_2[2], list_action)
    return action
#################################################################
@njit()
def agent_hieu_270922(state,file_temp,file_per):
    actions = getValidActions(state)
    actions = np.where(actions == 1)[0]
    action = np.random.choice(actions)
    file_per = (len(state),getActionSize())
    return action,file_temp,file_per

# LEN_STATE_hieu_270922,AMOUNT_ACTION_hieu_270922 = normal_main([agent_hieu_270922]*getAgentSize(), 1, [0])[1]

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
    norm_action = np.zeros(getActionSize())
    norm_action[list_action] = 1
    norm_action = norm_action.reshape(1, getActionSize())

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
    list_action = getValidActions(state)
    list_action = np.where(list_action == 1)[0]
    action = neural_network_hieu_270922(state, file_per_2[0], file_per_2[1], file_per_2[2], list_action)
    return action

######################################################################
######################################################################
######################################################################

@njit()
def file_temp_to_action_Phong_130922(state, file_temp):
    a = getValidActions(state)
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
def get_func(player_state, id, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10):
    if id == 0: return test2_An_130922(player_state, per0)
    elif id == 1: return test2_Dat_130922(player_state, per1)
    elif id == 2: return test2_Hieu_130922(player_state, per2)
    elif id == 3: return test2_Hieu_270922(player_state, per3)
    elif id == 4: return test2_Khanh_200922(player_state, per4)
    elif id == 5: return test2_NhatAnh_130922(player_state, per5)
    elif id == 6: return test2_NhatAnh_200922(player_state, per6)
    elif id == 7: return test2_NhatAnh_270922(player_state, per7)
    elif id == 8: return test2_Phong_130922(player_state, per8)
    elif id == 9: return test2_Phong_130922(player_state, per9)
    else: return test2_Phong_130922(player_state, per10)


@njit()
def one_game_numba(p0, list_other, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10):
    env = initEnv()
    _temp_ = List()
    _temp_.append(np.array([[0]]))

    winner = -1
    while env[230] < MAX_TURN_IN_ONE_GAME:
        idx = int(env[254])
        player_state = getAgentState(env)
        if list_other[idx] == -1:
            action, _temp_, per_player = p0(player_state,_temp_,per_player)
            # print('ac', action, type(action), 'acc')
        else:
            action = get_func(player_state, list_other[idx], per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10)
            # print('ac', action, type(action), 'acc')
        if getValidActions(player_state)[action] != 1:
            raise Exception('bot dua ra action khong hop le')
        stepEnv(env, action)

        winner = checkEnded(env)
        if winner != -1:
            break

    env[np.array([68, 110, 152, 194])] += env[np.array([67, 109, 151, 193])]
    if winner != -1:
        for i in range(4):
            env[254] = i
            env[229] = 2
            p_state = getAgentState(env)
            if list_other[i] == -1:
                act, _temp_, per_player = p0(p_state, _temp_, per_player)

    winner_ = False
    if np.where(list_other == -1)[0] ==  winner: winner_ = True
    else: winner_ = False
    return winner_,  per_player



@njit()
def n_game_numba(p0, num_game, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10):
    win = 0
    for _n in range(num_game):
        list_other = np.append(np.random.choice(np.arange(11), 3), -1)
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

@njit()
def one_game_numba_2(p0, list_other, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10):
    env = initEnv()
    _temp_ = List()
    _temp_.append(np.array([[0]]))

    winner = -1
    while env[230] < MAX_TURN_IN_ONE_GAME:
        idx = int(env[254])
        player_state = getAgentState(env)
        if list_other[idx] == -1:
            action, _temp_, per_player = p0(player_state,_temp_,per_player)
            # print('ac', action, type(action), 'acc')
        else:
            action = get_func(player_state, list_other[idx], per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10)
            # print('ac', action, type(action), 'acc')
        if getValidActions(player_state)[action] != 1:
            raise Exception('bot dua ra action khong hop le')
        stepEnv(env, action)

        winner = checkEnded(env)
        if winner != -1:
            break

    env[np.array([68, 110, 152, 194])] += env[np.array([67, 109, 151, 193])]
    if winner != -1:
        for i in range(4):
            env[254] = i
            env[229] = 2
            p_state = getAgentState(env)
            if list_other[i] == -1:
                act, _temp_, per_player = p0(p_state, _temp_, per_player)

    winner_ = False
    if np.where(list_other == -1)[0] ==  winner: winner_ = True
    else: winner_ = False
    return winner_,  per_player



@njit()
def n_game_numba_2(p0, num_game, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10):
    win = 0
    for _n in range(num_game):
        list_other = np.append(np.random.choice(np.arange(11), 3), -1)
        np.random.shuffle(list_other)
        winner,per_player  = one_game_numba_2(p0, list_other, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10)
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
    return n_game_numba_2(p0, n_game, per_player, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, per10)

