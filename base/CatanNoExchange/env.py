from base.CatanNoExchange.relation import *
import random

@njit()
def initEnv() -> np.ndarray:
    env = np.zeros(279)

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
    env[48:53] = 15

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
    env[229] = 0
    # [230]: Turn

    # [231]: Điểm đặt thứ nhất
    env[231] = -1

    # [232]: Số tài nguyên trả do bị chia

    # [233]: Đang dùng thẻ dev gì
    env[233] = -1

    # [234]: Số lần sử dụng thẻ dev
    # [235:239]: Loại thẻ dev được sử dụng trong turn hiện tại
    # [239:244]: Lượng nguyên liệu còn lại khi bị chia đầu game
    resource_turn_0 = np.random.randint(0, 5)
    env[239:244] = 4
    env[239 + resource_turn_0] = 3
    # [244]: Người chơi đang action (không hẳn là người chơi chính)
    # [245:249], số nguyên liệu đã lấy trong turn đầu game
    env[245] = 1 #Người chơi đầu tiên đã nhận 1 nguyên liệu
    # [249:254]: Tài nguyên đưa ra trong trade offer
    # [254:259]: Tài nguyên yêu cầu trong trade offer
    # [254:274]: Tài nguyên trong kho dự trữ của các người chơi
    env[254 + resource_turn_0] = 1
    return env


@njit()
def getAgentState(env: np.ndarray) -> np.ndarray:
    p_state = np.zeros(LEN_P_STATE)

    p_idx = int(env[244])
    turn = env[230]

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
    # [186]: 
    p_state[185:187] = env[228:230]
    p_state[186] = env[245+p_idx] #Nguyên liệu còn lại có thể lấy ở đầu game

    # [187]: Điểm đặt thứ nhất
    # [188]: Số tài nguyên phải bỏ do bị chia
    # [189]: Đang dùng thẻ dev gì
    # [190]: Số lần dùng thẻ dev
    # [191:195]: Loại thẻ dev được sử dụng trong turn hiện tại
    p_state[187:195] = env[231:239] 
    p_state[195:200] = env[239:244] #Số nguyên liệu còn lại ở trong kho
    p_state[200:205] = env[254+5*p_idx:259+5*p_idx]     #Các ngyên liệu có trong kho của người chơi
    phase = int(env[229])
    # [205:218]: Các phase, gồm 13 phase 0 -> 12, phase 12 là phase chọn lấy nguyên liệu từ kho
    p_state[205 + phase] = 1 
    p_state[218] = np.argmax(env[249:254]) #Tài nguyên đưa ra trong trade offer để trade với bank
    return p_state


@njit()
def getValidActions(p_state: np.ndarray):
    phase = np.where(p_state[205:205+13] == 1)[0]
    list_action = np.full(AMOUNT_ACTION, 0)
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

    if phase == 6:  # Chọn các mô đun giữa turn
        check_useDev(p_state, list_action)

        if (p_state[54:59] >= ROAD_PRICE).all():
            # Mua đường (83)
            if p_state[79] == -1 and check_firstPoint(p_state):
                list_action[83] = 1

            # Mua nhà (84)
            if (p_state[54:59] >= SETTLEMENT_PRICE).all() and p_state[84] == -1:
                p_point, all_stm_and_city = get_p_point_n_all_stm_city(p_state)
                list_point = np.where(p_point == 1)[0]
                for point in list_point:
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
                        list_action[84] = 1
                        break

        # Mua thành phố (85)
        if (p_state[54:59] >= CITY_PRICE).all() and p_state[80] != -1 and p_state[88] == -1:
            list_action[85] = 1

        # Mua thẻ dev (86)
        if (p_state[54:59] >= DEV_PRICE).all() and p_state[53] == 1:
            list_action[86] = 1

        if (p_state[54:59] > 0).any(): # Trade với bank (87)
            if (p_state[54:59] >= p_state[91:96]).any():
                temp = np.where(p_state[54:59] >= p_state[91:96])[0]
                for res in temp:
                    for res_1 in range(5):
                        if res_1 != res and p_state[48+res_1] == 1:
                            list_action[87] = 1
                            break

                    if list_action[87] == 1:
                        break

        # Action lấy nguyên liệu từ kho
        if (p_state[200:205] > 0).any() and p_state[186] == 1:
            # print('hi', p_state[200:205], (p_state[200:205] > 0).all())
            list_action[94] = 1

        # Kết thúc lượt (92)
        list_action[88] = 1

        return list_action

    if phase == 3:  # Trả tài nguyên do bị chia bài
        for i in range(5):
            if p_state[54+i] > 0:
                list_action[89+i] = 1

        return list_action

    if phase == 11:  # Chọn tài nguyên muốn nhận từ ngân hàng
        # Chọn những tài nguyên mà ngân hàng có, khác tài nguyên đưa ra
        list_action[59:64] = p_state[48:53]
        # list_action[59+np.argmax(p_state[196:201])] = 0
        list_action[int(59 + p_state[218])] = 0

        return list_action

    if phase == 10:  # Chọn tài nguyên khi trade với ngân hàng
        # Chọn những tài nguyên mà khi chọn, ngân hàng còn ít nhất 1 loại tài nguyên khác
        for res in range(5):
            if p_state[54+res] >= p_state[91+res]:
                for res_1 in range(5):
                    if res_1 != res and p_state[48+res_1] == 1:
                        list_action[89+res] = 1
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

    if phase == 5: # Lấy nguyên liệu đầu game
        list_action[59:64] = (p_state[195:200] > 0)*1

    if phase == 2:  # Đổ xx hoặc dùng thẻ dev
        # Đổ xx
        list_action[54] = 1

        check_useDev(p_state, list_action)

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

    if phase == 12: #Lấy nguyên liệu từ kho ở trước hoặc sau khi đổ xúc xắc
        list_action[59:64] = (p_state[200:205] > 0)*1
        return list_action
    return list_action

@njit()
def random_choice_weight(a):
    b = []
    for id_res, count in enumerate(a):
        for number_res in range(int(count)):
            b.append(id_res)
    res_choice = np.random.randint(0, len(b))
    return b[res_choice]

@njit()
def stepEnv(env: np.ndarray, action: int):
    phase = env[229]
    p_idx = int(env[244])
    s_ = 58 + 42*p_idx


    if phase == 6:  # Chọn các mô đun giữa turn
        if action == 88:  # Kết thúc lượt
            env[239] = 1  # Reset lại số lần tạo trade offer
            env[230] += 1  # Tăng turn
            n_idx = int(env[230] % 4)
            env[245+n_idx] = 1
            s_n = int(58 + 42*n_idx)
            env[244] = n_idx  # Người đi tiếp theo

            # Kiểm tra thẻ dev có thể sử dụng
            env[235:239] = env[s_n+5:s_n+9] > 0
            # print(env[235:239])

            # Tổng xx
            env[228] = -1
            # print('Trả lại giá trị cho xúc xắc và đổi người chơi', env[228])

            # Pha
            env[229] = 2

            if (env[235:239] == 0).all():
                roll_xx(env)
                # print('sau khi roll', env[228])

            return


        if action == 87:  # Trade với bank
            env[229] = 10

            return

        if action == 83:  # Mua đường
            env[s_:s_+5] -= ROAD_PRICE  # Trả tài nguyên
            env[48:53] += ROAD_PRICE
            env[229] = 1  # Chuyển sang pha đặt đường

            return

        if action == 86:  # Mua thẻ dev, không chuyển pha
            env[s_:s_+5] -= DEV_PRICE  # Trả tài nguyên mua thẻ dev
            env[48:53] += DEV_PRICE
            temp = env[53:58]
            # print('Các thẻ dev còn lại trong game', temp)
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

        if action == 84:  # Mua nhà
            env[s_:s_+5] -= SETTLEMENT_PRICE
            env[48:53] += SETTLEMENT_PRICE
            env[229] = 8  # Sang pha mua nhà

            return

        if action == 85:  # Mua thành phố
            env[s_:s_+5] -= CITY_PRICE
            env[48:53] += CITY_PRICE
            env[229] = 9  # Chuyển sang pha mua thành phố

            return

        if action == 94: #Lấy thẻ từ kho dự trữ phát triển
            env[229] = 12
            env[245:249] = 0
            return

    if phase == 3:  # Trả tài nguyên do bị chia bài
        env[s_+action-89] -= 1  # Trừ tài nguyên của người chơi
        env[action-41] += 1  # Trả tài nguyên này cho ngân hàng

        env[232] -= 1  # Giảm số lượng thẻ cần trả
        if env[232] == 0:  # Đã trả đủ
            for i in range(1, 5):
                e_idx = (p_idx + i) % 4
                s_e = 58 + 42*e_idx
                if e_idx == env[230] % 4:  # Quay về người chơi chính
                    env[229] = 4  # Sang pha di chuyển Robber
                    env[244] = e_idx  # Chuyển người action
                    break
                else:
                    if np.sum(env[s_e:s_e+5]) > 7:
                        # Cập nhật số lá phải bỏ
                        env[232] = np.sum(env[s_e:s_e+5]) // 2
                        env[244] = e_idx  # Chuyển người action
                        break

        return

    if phase == 11:  # Chọn tài nguyên muốn nhận từ ngân hàng
        res_idx = action - 59

        # Người chơi
        env[s_+res_idx] += 1
        env[s_:s_+5] -= env[249:254]

        # Ngân hàng
        env[48+res_idx] -= 1
        env[48:53] += env[249:254]

        env[249:254] = 0
        env[229] = 6

        return

    if phase == 10:  # Chọn tài nguyên khi trade với ngân hàng
        res_idx = action - 89
        env[249+res_idx] = env[s_+37+res_idx]

        env[229] = 11

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

        # 16 turn đầu: Kết thúc turn, chuyển người chơi, nếu là turn 8 thì chỉ chuyển pha
        if env[230] <= 15:
            env[230] += 2
            turn = env[230]
            if env[230] == 16:  # Chỉ chuyển pha
                env[239] = 1  # Cài lại số lần tạo trade offer
                env[229] = 2  # Sang pha 2: Đổ xx hoặc dùng thẻ dev
                roll_xx(env)  # Chưa thể có thẻ dev trong trường hợp này
            else:  # Thay đổi người chơi, chuyển pha
                env[229] = 0
                if env[230] < 8:
                    env[244] = env[230] // 2
                    total_resource_take = int(turn//2 + 1)
                    for i_res in range(total_resource_take):
                        # res = random.choices([0,1,2,3,4], weights = env[239:244])[0]
                        res = random_choice_weight(env[239:244])
                        env[239+res] -= 1
                        if p_idx != 3:
                            o_idx = p_idx+1
                        else: o_idx = 3
                        env[254+5*o_idx+res] += 1
                        env[245+o_idx] += 1
                else:
                    env[244] = 7 - env[230]//2
                    if turn == 8 :
                        o_idx = int(p_idx)
                        total_resource_take = int(5 - env[245 + o_idx])
                    else:
                        o_idx = int(p_idx - 1)
                        total_resource_take = int(5 - env[245 + o_idx])
                    for i_res in range(total_resource_take):
                        # res = random.choices([0,1,2,3,4], weights = env[239:244])[0]
                        res = random_choice_weight(env[239:244])
                        env[239+res] -= 1
                        env[254+5*o_idx+res] += 1
                        env[245+o_idx] += 1


        else:  # Các turn giữa game
            if env[233] != 1:  # Không dùng thẻ roadbuilding, về pha 6
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
        res = int(env[int(env[19])])
        if res != 5 and env[48+res] > 0:
            env[s_+res] += 1 #Người chơi
            env[48+res] -= 1 #Ngân hàng
        after_Rob(env)

        return

    if phase == 5:  # Lấy nguyên liệu đầu game
        res_idx = action - 59
        env[254+5*p_idx+res_idx] += 1
        env[239+res_idx] -= 1
        turn = env[230]
        env[245+p_idx] += 1
        if turn <=7:
            if env[245+p_idx] < (turn//2 + 1):
                env[229] = 5
            else:
                env[229] = 0 #Lấy nguyên liệu xong chuyển sang phase 0
                env[230] += 1
                env[244] = turn // 2
        elif turn <= 15:
            if env[245+p_idx] < 5:
                env[229] = 5
            else:
                env[229] = 0 #Lấy nguyên liệu xong chuyển sang phase 0
                env[230] += 1
                env[244] = (3 - (turn - 8)//2)
                # print('Chuyển người chơi thành, ', (3 - (turn - 8)//2), env[244])

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
        if env[230] == 15:
            env[245:249] = 0
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
            if env[48 + res_idx] > 0:
                res_thieu = 19 - env[48 + res_idx]
                if res_thieu > env[48 + res_idx]:
                    res_thieu = env[48 + res_idx]

                env[48 + res_idx] -= res_thieu
                env[s_+res_idx] += res_thieu
            # for i in range(1, 4):
            #     e_idx = (p_idx + i) % 4
            #     s_e = 58 + 42*e_idx
            #     sum_res += env[s_e+res_idx]
            #     env[s_e+res_idx] = 0

            # env[s_+res_idx] += sum_res
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

    if phase == 12: # Lấy nguyên liệu ở trong kho trước hoặc sau khi đổ xúc xắc
        res = action - 59 
        env[58 + 42*p_idx + res] += 1 #Cộng thêm nguyên liệu cho người chơi
        env[254 + 5*p_idx + res] -= 1 #Trừ nguyên liệu cho kho của người chơi
        env[245+p_idx] = 0
        if env[228] == -1:
            env[229] = 2
            roll_xx(env)
        else:
            env[229] = 6

        return 
    
    return

@njit()
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


@njit()
def check_firstPoint(p_state: np.ndarray):
    p_point, all_road = get_p_point_n_all_road(p_state)
    list_point = np.where(p_point == 1)[0]
    for point in list_point:
        for road in POINT_ROAD[point]:
            if road != -1 and all_road[road] == 0:
                return True

    return False


@njit()
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


@njit()
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



@njit()
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


@njit()
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


@njit()
def roll_xx(env: np.ndarray):
    p_idx = int(env[244])
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
                env[244] = e_idx  # Đổi người chơi action
                break

    else:  # Trả tài nguyên từ ngân hàng
        temp = np.where(env[20:39] == env[228])[0]
        list_tile = temp[temp != env[19]]
        # print('Ngân hàng trả tài nguyên', list_tile, env[list_tile])
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


@njit()
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


@njit()
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


@njit()
def after_useDev(env: np.ndarray):
    env[233] = -1
    env[234] = 0
    if env[228] == -1:  # Chưa roll xx
        env[229] = 2
        roll_xx(env)
    else:  # Đã roll xx
        env[229] = 6


@njit()
def after_Rob(env: np.ndarray):
    if env[233] == 0:  # Đang dùng thẻ knight
        after_useDev(env)
    else:  # Vừa đổ ra 7
        env[229] = 6


@njit()
def weighted_random(p: np.ndarray):
    a = np.sum(p)
    b = np.random.uniform(0, a)
    for i in range(len(p)):
        b -= p[i]
        if b <= 0:
            return i



@njit()
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


@njit()
def getReward(p_state: np.ndarray):
    if p_state[64] > 9:  # Hơn 9 điểm => auto thắng
        return 1

    elif p_state[98] > 9 or p_state[127] > 9 or p_state[156] > 9:
        return 0

    else:
        return -1  # Chưa có trên 9 điểm, chưa kết thúc game



@njit()
def getStateSize():
    return LEN_P_STATE


@njit()
def getActionSize():
    return AMOUNT_ACTION


@njit()
def getAgentSize():
    return 4


def one_game(list_player, per_file):
    env = initEnv()
    temp_file = [[0], [0], [0], [0]]

    winner = -1
    while env[230] < 1000:
        p_idx = int(env[244])
        p_state = getAgentState(env)
        actions = getValidActions(p_state)

        action, temp_file[p_idx], per_file = list_player[p_idx](
            p_state, temp_file[p_idx], per_file)

        if actions[action] != 1:
            raise Exception('Action không hợp lệ')

        stepEnv(env, action)


        winner = checkEnded(env)
        if winner != -1:
            break

    env[np.array([68, 110, 152, 194])] += env[np.array([67, 109, 151, 193])]

    if winner != -1:
        for i in range(4):
            env[244] = i
            env[229] = 2
            p_state = getAgentState(env)
            action, temp_file[i], per_file = list_player[i](
                p_state, temp_file[i], per_file)

    return winner, per_file


def normal_main(list_player, times, per_file):
    count_win = np.full(9, 0)
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



def numba_one_game(p_lst_idx_shuffle, p0, p1, p2, p3, per_file):
    env = initEnv()
    temp_file = [[0], [0], [0], [0]]

    temp_1_player = List()
    temp_1_player.append(np.array([[0.]]))
    temp_file = [temp_1_player]*(getAgentSize())

    winner = -1
    while env[230] < 1000:
        p_idx = int(env[244])
        p_state = getAgentState(env)
        actions = getValidActions(p_state)

        if p_lst_idx_shuffle[p_idx] == 0:
            act, temp_file[p_idx], per_file = p0(p_state, temp_file[p_idx], per_file)
        elif p_lst_idx_shuffle[p_idx] == 1:
            act, temp_file[p_idx], per_file = p1(p_state, temp_file[p_idx], per_file)
        elif p_lst_idx_shuffle[p_idx] == 2:
            act, temp_file[p_idx], per_file = p2(p_state, temp_file[p_idx], per_file)
        else:
            act, temp_file[p_idx], per_file = p3(p_state, temp_file[p_idx], per_file)

        if actions[act] != 1:
            raise Exception('Action không hợp lệ')

        stepEnv(env, act)

        winner = checkEnded(env)
        if winner != -1:
            break

    env[np.array([68, 110, 152, 194])] += env[np.array([67, 109, 151, 193])]

    if winner != -1:
        for i in range(4):
            env[244] = i
            env[229] = 2
            p_state = getAgentState(env)

            if p_lst_idx_shuffle[p_idx] == 0:
                act, temp_file[p_idx], per_file = p0(p_state, temp_file[p_idx], per_file)
            elif p_lst_idx_shuffle[p_idx] == 1:
                act, temp_file[p_idx], per_file = p1(p_state, temp_file[p_idx], per_file)
            elif p_lst_idx_shuffle[p_idx] == 2:
                act, temp_file[p_idx], per_file = p2(p_state, temp_file[p_idx], per_file)
            else:
                act, temp_file[p_idx], per_file = p3(p_state, temp_file[p_idx], per_file)

    return winner, per_file


def numba_main(p0, p1, p2, p3, times, per_file):
    count_win = np.full(9, 0)
    p_lst_idx = np.arange(4)

    for _n in range(times):
        np.random.shuffle(p_lst_idx)
        winner, per_file = numba_one_game(
            p_lst_idx, p0, p1, p2, p3, per_file
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


@njit()
def random_Env(p_state):
    arr_action = getValidActions(p_state)
    arr_action = np.where(arr_action == 1)[0]
    act_idx = np.random.randint(0, len(arr_action))
    return arr_action[act_idx]

@njit()
def one_game_numba(p0, list_other, per_player):
    env = initEnv()

    _temp_ = List()
    _temp_.append(np.array([[0]]))

    _cc = 0
    winner = -1
    while env[230] < 1000:
        p_idx = int(env[244])
        p_state = getAgentState(env)
        actions = getValidActions(p_state)

        if list_other[p_idx] == -1:
            action, _temp_, per_player = p0(p_state,_temp_,per_player)
        elif list_other[p_idx] == -2:
            action = random_Env(p_state)

        if actions[action] != 1:
            raise Exception('Action không hợp lệ')

        stepEnv(env, action)
        if checkEnded(env) != 0:
            break
        
        _cc += 1

    env[np.array([68, 110, 152, 194])] += env[np.array([67, 109, 151, 193])]

    if winner != -1:
        for p_idx in range(4):
            if list_other[p_idx] == -1:
                env[244] = p_idx
                env[229] = 2
                p_state = getAgentState(env)
                action, _temp_, per_player = p0(p_state,_temp_,per_player)
    if np.where(list_other == -1)[0] ==  (checkEnded(env)): winner = True
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
    return n_game_numba(p0, n_game, per_player, level)


def one_game_numba_2(p0, list_other, per_player):
    env = initEnv()

    _temp_ = List()
    _temp_.append(np.array([[0]]))

    _cc = 0
    winner = -1
    while env[230] < 1000:
        p_idx = int(env[244])
        p_state = getAgentState(env)
        actions = getValidActions(p_state)

        if list_other[p_idx] == -1:
            action, _temp_, per_player = p0(p_state,_temp_,per_player)
        elif list_other[p_idx] == -2:
            action = random_Env(p_state)

        if actions[action] != 1:
            raise Exception('Action không hợp lệ')

        stepEnv(env, action)
        if checkEnded(env) != 0:
            break
        
        _cc += 1

    env[np.array([68, 110, 152, 194])] += env[np.array([67, 109, 151, 193])]

    if winner != -1:
        for p_idx in range(4):
            if list_other[p_idx] == -1:
                env[244] = i
                env[229] = 2
                p_state = getAgentState(env)
                action, _temp_, per_player = p0(p_state,_temp_,per_player)
    if np.where(list_other == -1)[0] ==  (checkEnded(env)): winner = True
    else: winner = False
    return winner,  per_player


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
    return n_game_numba_2(p0, n_game, per_player, level)
