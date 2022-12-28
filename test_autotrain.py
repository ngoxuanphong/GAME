
from base.Splendor_v3.env import *
if len(sys.argv) >= 2:
    sys.argv = [sys.argv[0]]
sys.argv.append('Splendor_v3')

per_agent_env = np.array([0])
@njit()
def random_Env(p_state, per):
    arr_action = getValidActions(p_state)
    arr_action = np.where(arr_action == 1)[0]
    act_idx = np.random.randint(0, len(arr_action))
    return arr_action[act_idx], per

for i in range(3):
    win, per = numba_main_2(random_Env, 1000, per_agent_env, i)
    print(i, win)
