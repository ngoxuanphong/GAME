import base.TLMN.env as env
import base.TLMN_v2.env as env0
import base.MachiKoro.env as env1
import base.Splendor.env as env2
import base.Splendor_v2.env as env3
import base.Splendor_v3.env as env4
import base.Sheriff.env as env5
import base.SushiGo.env as env6

import base.StoneAge.env as env7
import base.Catan_v2.env as env8
import base.CatanNoExchange.env as env9
import base.Century.env as env10
from CheckEnv import check_env

# print(env, check_env(env))
# print(env0, check_env(env0))
# print(env1, check_env(env1))
# print(env2, check_env(env2))
# print(env3, check_env(env3))
# print(env4, check_env(env4))
# print(env5, check_env(env5))
# print(env6, check_env(env6))
# print(env7, check_env(env7))
# print(env8, check_env(env8))
# print(env9, check_env(env9))
# print(env10, check_env(env10))
from numba import njit
import numpy as np
@njit()
def test_numba(p_state, per_file):
    arr_action = env7.getValidActions(p_state)
    arr_action = np.where(arr_action == 1)[0]
    act_idx = np.random.randint(0, len(arr_action))
    if env7.getReward(p_state) != -1:
        per_file[0] += 1
    if env7.getReward(p_state) == 1:
        per_file[1] += 1
    return arr_action[act_idx], per_file

