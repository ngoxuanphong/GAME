
from base.Splendor_v2.env import *
@njit()
def test(p_state, temp_file, per_file):
    arr_action = getValidActions(p_state)
    arr_action = np.where(arr_action == 1)[0]
    act_idx = np.random.randint(0, len(arr_action))
    return arr_action[act_idx], temp_file, per_file

print('lv0', numba_main_2(test, 1000, [0], 0))
print('lv1', numba_main_2(test, 1000, [0], 1))
print('lv2', numba_main_2(test, 1000, [0], 2))
print('lv3', numba_main_2(test, 1000, [0], 3))