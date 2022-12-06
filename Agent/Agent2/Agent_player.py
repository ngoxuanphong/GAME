import sys, os
from setup import game_name
sys.path.append(os.path.abspath(f"base/{game_name}"))
from env import *

import numpy as np
def DataAgent():
    sonl = 5
    sola = 12
    w = np.zeros((getStateSize(),sonl*sola))
    # nl thiếu của thẻ đầu
    for idthe in range(sola):
    # idthe = 0
        for nl in range(sonl):
            w[18 + 7 * idthe + 2 + nl][idthe * 5 + nl] = -1
            w[6 + nl][idthe * 5 + nl] = 1
            w[12 + nl][idthe * 5 + nl] = 1

    w1 = np.zeros((sonl*sola,getActionSize()))
    # layer cuối, điều chỉnh action
    for idthe in range(sola):
        for nl in range(sonl):
            w1[idthe * 5 + nl][idthe] = 1
    return [w,w1]


@njit()
def Agent(state,temp,per):
    actions = getValidActions(state)
    output = state @ per[0]
    output = (output >= 0) *1.0
    output = output @ per[1]
    output = output * actions + actions
    action = np.argmax(output)
    return action,temp,per