from base.SushiGo.env import *

@njit()
def NA(state,temp,per):
    if per[1][0] == 0:
        index = np.argmax(per[3]/per[2])
        per[1][0] = 1
        per[1][1] = index
        if np.random.rand() < 0.05:
            replace = np.argmin(per[3]/per[2])
            per[0][replace] = np.random.rand(amount_action())
            per[2][replace] = 1
            per[3][replace] = 1
    index = int(per[1][1][0])
    mt = per[0][index]
    actions = get_list_action(state)
    output = mt * actions + actions
    action = np.argmax(output)
    # print(output,action,actions)
    win = check_victory(state)
    if win != -1:
        per[1][0] = 0
        per[2][index] += 1
        if win == 1:
            per[3][index] += 1
    return action,temp,per
per = [np.random.rand(100,amount_action()), # lưu trữ ma trận
       np.zeros((2,1)), #1: tình trạng ván chơi, 0 là trong game, 1 là index ma trận
       np.ones((100,1)), #2: số trận đã chơi
       np.ones((100,1)) #3: số trận đã thắng
        ]

    
import time
a = time.time()
win, x = numba_main_2(NA, per, 1)
b = time.time()
print('thời gian load game', b - a)

a = time.time()
sotran = 1000000
win, x = numba_main_2(NA, per, sotran)
b = time.time()
print('Thời gian chạy', b - a, win)