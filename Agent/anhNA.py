from numba.typed import List, Dict
import numpy as np
import sys, os
# game_name = sys.argv[1]
# sys.path.append(os.path.abspath(f"base/{sys.argv[1]}"))
# from env import *
from base.TLMN.env import *

# @njit()
# def DataAgent():
#     perx = Dict()
#     temp = List([np.zeros(getStateSize())-1])
#     temp.pop(0)
#     perx[0] = temp.copy() # ingame states when act == target
#     perx[1] = temp.copy() # thắng/thua
#     perx[2] = temp.copy()
#     perx[3] = temp.copy()
#     perx[4] = temp.copy()
#     perx[3].append(np.zeros(1)) # đếm số state
#     perx[4].append(np.zeros(1))
#     return perx

# @njit
# def Agent(state, per):
#     target_act = int(per[4][0][0])
#     actions = getValidActions(state)
#     output = np.random.rand(getActionSize()) * actions + actions
#     action = np.argmax(output)
#     win = getReward(state)
#     if win != -1:
#         if win == 1:
#             for times in range(int(per[3][0][0])):
#                 per[1].append(np.ones(1))
#         if win == 0:
#             for times in range(int(per[3][0][0])):
#                 per[1].append(np.zeros(1))
#         per[3][0][0] = 0 
#     return action, per
data = np.array([0])
def p0(state, per):
    actions = getValidActions(state)
    actions = np.where(actions == 1)
    id = np.random.randint(0, len(actions))
    return actions[id]


win, per = numba_main_2(p0,10000000, data, 0)

