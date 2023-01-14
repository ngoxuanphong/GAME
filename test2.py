import numpy as np
from base.Splendor_v2.env import *
per_find_value = [np.zeros(3),np.zeros(getStateSize()),
                np.zeros(getStateSize()),
                np.zeros(getStateSize())]

@njit()
def agent_find_value(state,per):
    actions = getValidActions(state)
    output = np.random.rand(getActionSize()) * actions + actions
    action = np.argmax(output)
    win = getReward(state)
    if per[0][0] < 1000:
        if win == 1:
            per[1] = np.minimum(per[1],state)
            per[2] = np.maximum(per[2],state)
            per[3] = state
            per[0][0] += 1
    else:
        value = np.zeros(getStateSize())
        for id in range(getStateSize()):
            sample = np.zeros(getStateSize()) + per[3]
            ifmin, ifmax = 0,0
            sample[id] = per[1][id]
            if getReward(sample) == 1:
                ifmin = 1
            sample[id] = per[2][id]
            if getReward(sample) == 1:
                ifmax = 1
            if ifmax > ifmin:
                value[id] = 1
            if ifmin > ifmax:
                value[id] = -1
        per[1] = value
    per[0][1] += 1
    if win != -1:
        if per[0][2] == 0:
            per[0][2] = per[0][1]
        else:
            per[0][2] = np.maximum(per[0][1],per[0][2])
        per[0][1] = 0
    return action, per

a, per_find_value = numba_main_2(agent_find_value,200000,per_find_value,0)
a