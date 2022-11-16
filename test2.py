
from base.Splendor_v3.env import *
@njit()
def test(p_state, temp_file, per_file):
    arr_action = getValidActions(p_state)
    arr_action = np.where(arr_action == 1)[0]
    act_idx = np.random.randint(0, len(arr_action))
    return arr_action[act_idx], temp_file, per_file

normal_main([test]*getAgentSize(), 1000, [0])