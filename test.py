from base.Splendor_v2.env_intern import *


# @njit()
def random_player(p_state):
    arr_action = getValidActions(p_state)
    arr_action = np.where(arr_action == 1)[0]
    print(arr_action)

    board_stock = p_state[:6]
    p_stock =  p_state[6:12]
    p_count_stock = p_state[12:17]

    for action in arr_action: #Nếu có action lấy thẻ thì lấy thẻ luôn
        if action in np.arange(1, 16):
            return action
    
    for action in arr_action: #Nếu có nguyên liệu thì lấy nguyên liệu
        if action in np.arange(31, 36):
            return action
    
    act_idx = np.random.randint(0, len(arr_action))
    return arr_action[act_idx]

print(intern_main(random_player, 1, True))

# import numpy as np
# print(np.arange(1, 13))
