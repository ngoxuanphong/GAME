import base.TLMN_v2.env as env0
import base.MachiKoro.env as env1
import base.TLMN.env as env2
import base.Splendor_v2.env as env3
import base.Splendor_v3.env as env4
import base.Sheriff.env as env5
import base.SushiGo.env as env6

import base.StoneAge.env as env7
import base.Catan_v2.env as env8
import base.Century.env as env9
from CheckEnv import check_env

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

import base.StoneAge.env_new as env_new
# print(env, check_env(env))

import numpy as np
for i in range(10000):
    p_state1 = np.random.randint(-100, 100, env_new.getStateSize())
    p_state2 = np.random.randint(0, 1000, env_new.getStateSize())
    p_state3 = np.random.randn(10)*100
    env_new.getValidActions(p_state1)
    env_new.getValidActions(p_state2)
    env_new.getValidActions(p_state3)
    env_new.getReward(p_state1)
    env_new.getReward(p_state2)
    env_new.getReward(p_state3)
