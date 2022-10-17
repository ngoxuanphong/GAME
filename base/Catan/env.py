import sys
import numpy as np
from numba import njit
import numpy as np

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


@njit
def reset():
    env = np.zeros(255)

    # [0:19]: Tài nguyên trên các ô đất
    temp = np.array([5, 0, 0, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 1, 1, 1, 4, 4, 4])
    np.random.shuffle(temp)
    env[0:19] = temp

    # [19]: Ví trí Robber
    env[19] = np.where(temp == 5)[0][0]

    # [20:39]: Số trên các ô đất
    temp_prob = np.zeros(19)
    temp = np.ones(19)
    temp[int(env[19])] = 0
    temp_1 = temp.copy()
    for i in range(4):
        k = np.random.choice(np.where(temp == 1)[0])
        temp[k] = 0
        temp_1[k] = 0
        if i < 2:
            temp_prob[k] = 6
        else:
            temp_prob[k] = 8

        temp[TILE_TILE[k][TILE_TILE[k] != -1]] = 0

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
def get_player_state(env: np.ndarray):
    p_state = np.zeros(210)

    p_idx = int(env[254])
    if env[230] >= 4 and env[230] <= 7:
        main_p_idx = 7 - env[230]
    else:
        main_p_idx = env[230] % 4

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
        if env[226+i] == -1:
            p_state[183+i] = -1
        else:
            p_state[183+i] = (env[226+i] - p_idx) % 4

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

    temp = p_state[65:80]
    list_road = temp[temp != -1].astype(np.int32)
    all_road[list_road] = 1
    p_point[ROAD_POINT[list_road].flatten()] = 1

    for i in range(3):
        s_ = 96 + 29*i

        temp = p_state[s_+3:s_+18]
        list_road = temp[temp != -1].astype(np.int32)
        all_road[list_road] = 1

        temp = p_state[s_+18:s_+27]
        stm_and_city = temp[temp != -1].astype(np.int32)
        p_point[stm_and_city] = 0

    return p_point, all_road


@njit
def check_firstPoint(p_state: np.ndarray):
    p_point, all_road = get_p_point_n_all_road(p_state)
    list_point = np.where(p_point == 1)[0]
    for point in list_point:
        list_road = POINT_ROAD[point][POINT_ROAD[point] != -1]
        if (all_road[list_road] == 0).any():
            return True

    return False


@njit
def check_useDev(p_state: np.ndarray, list_action: np.ndarray):
    # Knight: Có là được dùng
    if p_state[191] == 1:
        list_action[55] = 1

    # Roadbuilding: Còn đường và còn vị trí xây đường
    if p_state[192] == 1 and (p_state[65:80] == -1).any() and check_firstPoint(p_state):
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

    temp = p_state[65:80]
    list_road = temp[temp != -1].astype(np.int32)
    p_point[ROAD_POINT[list_road].flatten()] = 1

    temp = p_state[80:89]
    stm_and_city = temp[temp != -1].astype(np.int32)
    all_stm_and_city[stm_and_city] = 1
    p_point[stm_and_city] = 0

    for i in range(3):  # Loại đi các điểm của đối thủ
        s_ = 96 + 29*i
        temp = p_state[s_+18:s_+27]
        stm_and_city = temp[temp != -1].astype(np.int32)
        all_stm_and_city[stm_and_city] = 1
        p_point[stm_and_city] = 0

    return p_point, all_stm_and_city

@njit
def amount_action():
    return 106

@njit
def amount_state():
    return 210

@njit
def amount_player():
    return 4

@njit
def get_list_action(p_state: np.ndarray):
    phase = p_state[186]
    list_action = np.zeros(106)

    if phase == 0:  # Chọn điểm đặt nhà đầu game
        list_action[0:54] = 1

        temp = p_state[np.array([80, 81, 114, 115, 143, 144, 172, 173])]
        all_stm = temp[temp != -1].astype(np.int32)
        list_action[all_stm] = 0
        for stm in all_stm:
            temp = POINT_POINT[stm][POINT_POINT[stm] != -1]
            list_action[temp] = 0

    elif phase == 1:  # Chọn các điểm đầu mút của đường
        if p_state[187] == -1:  # Chọn điểm thứ nhất
            p_point, all_road = get_p_point_n_all_road(p_state)
            list_point = np.where(p_point == 1)[0]
            for point in list_point:
                list_road = POINT_ROAD[point][POINT_ROAD[point] != -1]
                if (all_road[list_road] == 0).any():
                    list_action[point] = 1

        else:  # Chọn điểm thứ hai
            all_road = np.zeros(72)
            temp = p_state[65:80]
            list_road = temp[temp != -1].astype(np.int32)
            all_road[list_road] = 1

            for i in range(3):
                s_ = 96 + 29*i
                temp = p_state[s_+3:s_+18]
                list_road = temp[temp != -1].astype(np.int32)
                all_road[list_road] = 1

            first_point = int(p_state[187])
            list_road = POINT_ROAD[first_point][POINT_ROAD[first_point] != -1]
            for road in list_road:
                if all_road[road] == 0:
                    list_action[ROAD_POINT[road]] = 1

            list_action[first_point] = 0

    elif phase == 2:  # Đổ xx hoặc dùng thẻ dev
        # Đổ xx
        list_action[54] = 1

        check_useDev(p_state, list_action)

    elif phase == 3:  # Trả tài nguyên do bị chia bài
        list_action[95:100] = p_state[54:59] > 0

    elif phase == 4:  # Di chuyển Robber
        list_action[64:83] = 1
        list_action[int(64+p_state[19])] = 0

    elif phase == 5:  # Chọn người để cướp tài nguyên
        robber_pos = int(p_state[19])
        for i in range(3):
            s_ = 96 + 29*i
            if p_state[s_] > 0:  # Chỉ xét khi có tài nguyên
                temp = p_state[s_+18:s_+27]
                stm_and_city = temp[temp != -1].astype(np.int32)
                for point in stm_and_city:
                    if point in TILE_POINT[robber_pos]:
                        list_action[83+i] = 1
                        break

    elif phase == 6:  # Chọn các mô đun giữa turn
        check_useDev(p_state, list_action)

        if (p_state[54:59] >= ROAD_PRICE).all():
            # Mua đường (86)
            if (p_state[65:80] == -1).any() and check_firstPoint(p_state):
                list_action[86] = 1

            # Mua nhà (87)
            if (p_state[54:59] >= SETTLEMENT_PRICE).all() and (p_state[80:85] == -1).any():
                p_point, all_stm_and_city = get_p_point_n_all_stm_city(p_state)
                list_point = np.where(p_point == 1)[0]
                for point in list_point:
                    list_road = POINT_ROAD[point][POINT_ROAD[point] != -1]
                    nearest_points = ROAD_POINT[list_road].flatten()
                    if (all_stm_and_city[nearest_points] == 0).all():
                        list_action[87] = 1
                        break

        # Mua thành phố (88)
        if (p_state[54:59] >= CITY_PRICE).all() and (p_state[80:85] != -1).any() and (p_state[85:89] == -1).any():
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
                temp_1 = np.arange(5)
                for res in temp:
                    temp_2 = temp_1[temp_1 != res]
                    if (p_state[48+temp_2] == 1).any():
                        list_action[91] = 1
                        break

        # Kết thúc lượt (92)
        list_action[92] = 1

    elif phase == 7:  # Chọn tài nguyên khi dùng thẻ dev
        if p_state[189] == 2:  # Đang dùng yearofplenty
            list_action[59:64] = p_state[48:53]
        elif p_state[189] == 3:  # Đang dùng monopoly
            list_action[59:64] = 1

    elif phase == 8:  # Chọn các điểm mua nhà
        p_point, all_stm_and_city = get_p_point_n_all_stm_city(p_state)
        list_point = np.where(p_point == 1)[0]
        for point in list_point:
            list_road = POINT_ROAD[point][POINT_ROAD[point] != -1]
            nearest_points = ROAD_POINT[list_road].flatten()
            if (all_stm_and_city[nearest_points] == 0).all():
                list_action[point] = 1

    elif phase == 9:  # Chọn các điểm mua thành phố
        temp = p_state[80:85]
        p_stm = temp[temp != -1].astype(np.int32)
        list_action[p_stm] = 1

    elif phase == 10:  # Đưa ra tài nguyên khi trade với người
        # Nếu đã có ít nhất 1 tài nguyên, thì phải có action dừng
        if (p_state[196:201] > 0).any():
            list_action[103] = 1

        # Các action thêm tài nguyên: các tài nguyên mà bản thân có
        list_action[95:100] = p_state[54:59] > p_state[196:201]

        # Nếu số loại tài nguyên bỏ vào là 4 thì không cho bỏ loại thứ 5 vào
        if np.count_nonzero(p_state[196:201] > 0) == 4:
            list_action[95+np.where(p_state[196:201] == 0)[0][0]] = 0

    elif phase == 11:  # Yêu cầu tài nguyên khi trade với người
        # Nếu đã có ít nhất 1 tài nguyên, thì phải có action dừng
        if (p_state[201:206] > 0).any():
            list_action[104] = 1

        # Các action thêm tài nguyên: các loại tài nguyên mà không có trong phần đưa ra
        list_action[59:64] = p_state[196:201] == 0

    elif phase == 12:  # Người chơi phụ phản hồi trade
        # Action từ chối: 93, Action: đồng ý: 94
        list_action[93:95] = 1

        # Vào pha này thì chắc chắn người chơi phụ phải có thể trade
        # if (p_state[54:59] >= p_state[196:201]).all():
        #     list_action[94] = 1

    elif phase == 13:  # Người chơi chính duyệt trade
        # Action bỏ qua
        list_action[105] = 1

        # Chọn người để trade
        # Vào pha này thì chắc chắn có ít nhất một người đồng ý trade
        list_action[100:103] = p_state[206:209]

    elif phase == 14:  # Chọn tài nguyên khi trade với ngân hàng
        # Chọn những tài nguyên mà khi chọn, ngân hàng còn ít nhất 1 loại tài nguyên khác
        temp = np.where(p_state[54:59] >= p_state[91:96])[0]
        temp_1 = np.arange(5)
        for res in temp:
            temp_2 = temp_1[temp_1 != res]
            if (p_state[48+temp_2] == 1).any():
                list_action[95+res] = 1

    elif phase == 15:  # Chọn tài nguyên muốn nhận từ ngân hàng
        # Chọn những tài nguyên mà ngân hàng có, khác tài nguyên đưa ra
        list_action[59:64] = p_state[48:53]
        list_action[59+np.where(p_state[196:201] != 0)[0][0]] = 0

    list_action = np.where(list_action == 1)[0]
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
def step(env: np.ndarray, action: int):
    phase = env[229]
    p_idx = int(env[254])
    s_ = 58 + 42*p_idx

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

    elif phase == 1:  # Chọn các điểm đầu mút của đường
        if env[231] == -1:  # Chưa có điểm đặt thứ nhất
            env[231] = action

        else:
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
                if env[233] == 1:  # Đang dùng thẻ roadbuilding
                    env[234] -= 1
                    # Hoặc là hết lượt dùng, hoặc đủ 15 đường, hoặc hết vị trí xây đường
                    check = True
                    if env[234] == 0 or (env[s_+11:s_+26] != -1).all():
                        check = False

                    if check:
                        check = False
                        p_point = np.zeros(54)
                        all_road = np.zeros(72)

                        temp = env[s_+11:s_+26]  # Đường và điểm của người chơi
                        list_road = temp[temp != -1].astype(np.int32)
                        all_road[list_road] = 1
                        p_point[ROAD_POINT[list_road].flatten()] = 1

                        for i in range(1, 4):
                            e_idx = (p_idx + i) % 4
                            s_e = 58 + 42*e_idx

                            temp = env[s_e+11:s_e+26]
                            list_road = temp[temp != -1].astype(np.int32)
                            all_road[list_road] = 1

                            temp = env[s_e+26:s_e+35]
                            stm_and_city = temp[temp != -1].astype(np.int32)
                            p_point[stm_and_city] = 0

                        list_point = np.where(p_point == 1)[0]
                        for point in list_point:
                            nearest_roads = POINT_ROAD[point][POINT_ROAD[point] != -1]
                            if (all_road[nearest_roads] == 0).any():
                                check = True
                                break

                    if not check:  # Kết thúc trạng thái sử dụng thẻ roadbuilding
                        after_useDev(env)
                else:  # Không dùng thẻ roadbuilding, về pha 6
                    env[229] = 6

    elif phase == 2:  # Đổ xx hoặc dùng thẻ dev
        if action == 54:  # Đổ xúc xắc
            roll_xx(env)

        else:
            useDev(env, action, s_, p_idx)

    elif phase == 3:  # Trả tài nguyên do bị chia bài
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

    elif phase == 4:  # Di chuyển Robber
        env[19] = action - 64  # Cập nhật vị trí Robber

        # Xét xem có cướp được tài nguyên của ai hay ko
        check = False
        for i in range(4):
            if i != p_idx:
                s_i = 58 + 42*i
                if (env[s_i:s_i+5] > 0).any():  # Nếu người này có tài nguyên
                    temp = env[s_i+26:s_i+35]
                    stm_and_city = temp[temp != -1].astype(np.int32)
                    for point in stm_and_city:
                        if point in TILE_POINT[int(env[19])]:
                            env[229] = 5
                            check = True
                            break

            if check:
                break

        if not check:  # Không cướp được của ai
            after_Rob(env)

    elif phase == 5:  # Chọn người để cướp tài nguyên
        e_idx = (p_idx + action - 82) % 4
        s_e = 58 + 42*e_idx
        temp = env[s_e:s_e+5]  # Tài nguyên

        res_idx = weighted_random(temp)
        env[s_e+res_idx] -= 1  # Trừ tài nguyên của người bị cướp
        env[s_+res_idx] += 1  # Cộng tài nguyên cho người cướp

        after_Rob(env)
        return res_idx

    elif phase == 6:  # Chọn các mô đun giữa turn
        if action >= 55 and action < 59:  # Dùng thẻ dev
            useDev(env, action, s_, p_idx)

        elif action == 86:  # Mua đường
            env[s_:s_+5] -= ROAD_PRICE  # Trả tài nguyên
            env[48:53] += ROAD_PRICE
            env[229] = 1  # Chuyển sang pha đặt đường

        elif action == 87:  # Mua nhà
            env[s_:s_+5] -= SETTLEMENT_PRICE
            env[48:53] += SETTLEMENT_PRICE
            env[229] = 8  # Sang pha mua nhà

        elif action == 88:  # Mua thành phố
            env[s_:s_+5] -= CITY_PRICE
            env[48:53] += CITY_PRICE
            env[229] = 9  # Chuyển sang pha mua thành phố

        elif action == 89:  # Mua thẻ dev, không chuyển pha
            env[s_:s_+5] -= DEV_PRICE  # Trả tài nguyên mua thẻ dev
            env[48:53] += DEV_PRICE
            temp = env[53:58]

            dev_idx = weighted_random(temp)
            env[53+dev_idx] -= 1  # Trừ thẻ dev ở ngân hàng
            env[s_+5+dev_idx] += 1  # Cộng thẻ dev cho người chơi

            # Nếu thẻ dev vừa nhận là thẻ vp (dev_idx = 4) thì cộng điểm cho người chơi
            # if dev_idx == 4:
            #     env[s_+10] += 1
            # Note: Không được cộng do như thế các người khác sẽ biết

        elif action == 90:  # Trade với người
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

        elif action == 91:  # Trade với bank
            env[229] = 14

        elif action == 92:  # Kết thúc lượt
            env[239] = 1  # Reset lại số lần tạo trade offer
            env[230] += 1  # Tăng turn
            n_idx = env[230] % 4
            s_n = int(58 + 42*n_idx)
            env[254] = n_idx  # Người đi tiếp theo

            # Kiểm tra thẻ dev có thể sử dụng
            env[235:239] = env[s_n+5:s_n+9] > 0

            # Tổng xx
            env[228] = -1

            # Pha
            env[229] = 2

            if (env[235:239] == 0).all():
                roll_xx(env)

    elif phase == 7:  # Chọn tài nguyên khi dùng thẻ dev
        if env[233] == 2:  # Đang dùng thẻ yearofplenty
            env[action-11] -= 1  # Trừ tài nguyên cho ngân hàng
            env[s_+action-59] += 1  # Cộng tài nguyên cho người chơi

            # Giảm số lần sử dụng thẻ xuống
            env[234] -= 1

            # Nếu hết số lần sử dụng hoặc ngân hàng hết tài nguyên
            if env[234] == 0 or (env[48:53] == 0).all():
                after_useDev(env)

        elif env[233] == 3:  # Đang dùng thẻ monopoly
            sum_res = 0
            res_idx = action - 59
            for i in range(1, 4):
                e_idx = (p_idx + i) % 4
                s_e = 58 + 42*e_idx
                sum_res += env[s_e+res_idx]
                env[s_e+res_idx] = 0

            env[s_+res_idx] += sum_res
            after_useDev(env)

    elif phase == 8:  # Chọn các điểm mua nhà
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

    elif phase == 9:  # Chọn các điểm mua thành phố
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

    elif phase == 10:  # Đưa ra tài nguyên khi trade với người
        if action == 103:  # Kết thúc
            env[229] = 11  # Chuyển sang pha yêu cầu tài nguyên

        else:  # Thêm tài nguyên
            res_idx = action - 95
            env[240+res_idx] += 1
            if np.sum(env[240:245]) == np.sum(env[s_:s_+5]):  # Đã bỏ hết tài nguyên
                env[229] = 11

    elif phase == 11:  # Yêu cầu tài nguyên khi trade với người
        check = False
        if action == 104:  # Kết thúc
            check = True
        else:  # Thêm tài nguyên
            res_idx = action - 59
            env[245+res_idx] += 1
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

    elif phase == 12:  # Người chơi phụ phản hồi trade
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

    elif phase == 13:  # Người chơi chính duyệt trade
        if action == 105:  # Bỏ qua hết
            env[229] = 6  # Về luôn phase chọn mô đun
            env[240:250] = 0  # Reset trade
            env[251:254] = -1  # Reset phản hồi người chơi phụ
        else:
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

    elif phase == 14:  # Chọn tài nguyên khi trade với ngân hàng
        res_idx = action - 95
        env[240+res_idx] = env[s_+37+res_idx]

        env[229] = 15

    elif phase == 15:  # Chọn tài nguyên muốn nhận từ ngân hàng
        res_idx = action - 59

        # Người chơi
        env[s_+res_idx] += 1
        env[s_:s_+5] -= env[240:245]

        # Ngân hàng
        env[48+res_idx] -= 1
        env[48:53] += env[240:245]

        env[240:245] = 0
        env[229] = 6


@njit
def close_game(env: np.ndarray):
    score = env[np.array([68, 110, 152, 194])] + \
        env[np.array([67, 109, 151, 193])]
    max_score = np.max(score)
    if max_score >= 10:
        return np.where(score == max_score)[0][-1]

    return -1


def rdb(p_state, temp_file, per_file):
    arr_action = get_list_action(p_state)
    arr_action = np.where(arr_action == 1)[0]
    action = np.random.choice(arr_action)
    return action, temp_file, per_file


@njit
def rdb_numba(p_state, temp_file, per_file):
    arr_action = get_list_action(p_state)
    arr_action = np.where(arr_action == 1)[0]
    action = np.random.choice(arr_action)
    return action, temp_file, per_file


def o_n_e_g_a_m_e(list_player, per_file):
    env = reset()
    temp_file = [np.array([0]), np.array([0]), np.array([0]), np.array([0])]

    while env[230] < 1000:
        p_idx = int(env[254])
        p_state = get_player_state(env)
        action, temp_file[p_idx], per_file = list_player[p_idx](
            p_state, temp_file[p_idx], per_file)
        step(env, action)
        if close_game(env) != -1:
            break

    winner = close_game(env)
    if winner != -1:
        for i in range(4):
            env[254] = i
            env[229] = 2
            p_state = get_player_state(env)
            action, temp_file[i], per_file = list_player[i](
                p_state, temp_file[i], per_file)

    return winner, per_file, env[230]


def one_game(list_player, per_file):
    env = reset()
    temp_file = [[0], [0], [0], [0]]

    while env[230] < 1000:
        p_idx = int(env[254])
        p_state = get_player_state(env)
        action, temp_file[p_idx], per_file = list_player[p_idx](
            p_state, temp_file[p_idx], per_file)
        step(env, action)
        if close_game(env) != -1:
            break

    winner = close_game(env)
    if winner != -1:
        for i in range(4):
            env[254] = i
            env[229] = 2
            p_state = get_player_state(env)
            action, temp_file[i], per_file = list_player[i](
                p_state, temp_file[i], per_file)

    return winner, per_file


@njit
def update_turn_stats(stats: np.ndarray, value: float):
    if value == 1000:
        stats[4] += 1
    else:
        stats[1] = (stats[1] * stats[0] + value) / (stats[0] + 1)
        stats[0] += 1
        if value < stats[2]:
            stats[2] = value

        if value > stats[3]:
            stats[3] = value


def n_o_r_m_a_l_m_a_i_n(list_player, times, per_file):
    count_win = np.zeros(5)
    p_lst_idx = np.arange(4)
    turn_stats = np.array(
        [
            0.0,  # Số trận đã đấu
            0.0,  # Số turn trung bình
            999.0,  # Số turn min
            0.0,  # Số turn max
            0.0  # Số trận có turn = 1000
        ])

    for _n in range(times):
        if _n % 50 == 0 and _n != 0:
            print(_n, count_win, turn_stats)

        np.random.shuffle(p_lst_idx)
        winner, per_file, count_turn = o_n_e_g_a_m_e(
            [list_player[p_lst_idx[0]], list_player[p_lst_idx[1]],
                list_player[p_lst_idx[2]], list_player[p_lst_idx[3]]], per_file
        )

        update_turn_stats(turn_stats, count_turn)

        if winner == -1:
            count_win[4] += 1
        else:
            count_win[winner] += 1

    print(_n+1, count_win, turn_stats)
    return count_win, per_file, turn_stats


def normal_main(list_player, times, per_file):
    count_win = np.zeros(5)
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
            count_win[winner] += 1

    return count_win, per_file