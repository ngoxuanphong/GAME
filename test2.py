# from base.StoneAge.env import *
# import time

# def calculate_time(func):
#     def inner1(*args, **kwargs):
#         start = time.time()
#         func(*args, **kwargs)
#         end = time.time()
#         print('| Time to run code', end - start)
#     return inner1

# @njit()
# def test(p_state, per_file):
#     arr_action = getValidActions(p_state)
#     arr_action = np.where(arr_action == 1)[0]
#     act_idx = np.random.randint(0, len(arr_action))
#     check = getReward(p_state)
#     if check == 0:
#         per_file[0] += 1
#     if check == 1:
#         per_file[1] += 1
#     if check != -1:
#         per_file[2] += 1
#     return arr_action[act_idx], per_file

# @calculate_time
# def main():
#     a, per = normal_main([test]*getAgentSize(), 100000, np.array([0, 0, 0]))
#     print(a, per)
#     print('Check tổng số trận', (per[0] + per[1])/getAgentSize() == 100000)
#     print('Check số trận kết thúc', per[2] == (100000*getActionSize()))
#     print('Check tổng số trận thắng', sum(a[:-1]) == per[1]) #Có thể có 2 người thắng
#     print('Check tổng số trận thua', a[-1]*getAgentSize() == per[0]) #Có thể có nhiều người thua
    
# main()
import gym