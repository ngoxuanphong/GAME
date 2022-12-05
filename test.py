from base.CatanNoExchange.env import *
import time

@njit()
def random_player(p_state, temp_file, per_file):
    arr_action = getValidActions(p_state)
    arr_action = np.where(arr_action == 1)[0]
    id_act = np.random.randint(0, len(arr_action))
    return arr_action[id_act], temp_file, per_file

# @njit()
def random_player2(p_state, temp_file, per_file):
    arr_action = getValidActions(p_state)
    arr_action = np.where(arr_action == 1)[0]
    id_act = np.random.randint(0, len(arr_action))
    return arr_action[id_act], temp_file, per_file

a = time.time()
numba_main(random_player, random_player, random_player, random_player, 10000, np.array([0]))
b = time.time()
print(b - a)

numba_main_2(random_player, 5, np.array([0]), 0)
a = time.time()
numba_main_2(random_player, 10000, np.array([0]), 0)
b = time.time()
print(b - a)

a = time.time()
normal_main_2(random_player2, 10000, np.array([0]), 0)
b = time.time()
print(b - a)