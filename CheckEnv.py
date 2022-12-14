from base.Splendor.env import *

def CheckAllFunc():
    for func in [getActionSize,getActionSize,getStateSize,getAgentSize,getReward,getValidActions,normal_main,normal_main_2,numba_main,numba_main_2]:
        try:
            pass
        except:
            print('BUG')


def CheckReturn():
    for func in [getActionSize(),getStateSize(),getAgentSize()]:
        try:
            out = func
            if type(out) != int and type(out) != np.int64:
                print('BUG ')
        except:
            pass

@njit()
def test(p_state, temp_file, per_file):
    arr_action = getValidActions(p_state)
    arr_action = np.where(arr_action == 1)[0]
    act_idx = np.random.randint(0, len(arr_action))
    return arr_action[act_idx], temp_file, per_file

def CheckReturnRunGame():
    pass
def CheckRunGame():
    win, per = numba_main_2(test, 10000, [0], 0)
    win, per = normal_main_2(test, 10000, [0], 0)
    win, per = normal_main([test]*getAgentSize(), 10000, per)

CheckAllFunc()
CheckReturn()