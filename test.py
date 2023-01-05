from base.Catan_v2.env import *
import time
import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning,NumbaExperimentalFeatureWarning, NumbaWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)
warnings.simplefilter('ignore', category=NumbaWarning)

def calculate_time(func):
    def inner1(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print('| Time to run code', end - start)
    return inner1

# @jit()
def test(p_state, per_file):
    arr_action = getValidActions(p_state)
    arr_action = np.where(arr_action == 1)[0]
    act_idx = np.random.randint(0, len(arr_action))
    return arr_action[act_idx], per_file

def main():
    @calculate_time
    def test_normal_main():
        a, _ = normal_main([test]*getAgentSize(), 1000, np.array([0]))
        print('Normal_main', a, end = '')

    @calculate_time
    def test_numba_main():
        b, _ = numba_main(test, test, test, test, 1000, np.array([0]))
        print('Numba_main', b, end = '')

    @calculate_time
    def test_numba_main_2():
        c, _ = numba_main_2(test, 1000, np.array([0]), 0)
        print("numba_main_2", c, end = '')

    @calculate_time
    def test_normal_main_2():
        d, _ = normal_main_2(test, 1000, np.array([0]), 0)
        print('normal_main_2', d, end = '')
    
    # test_normal_main()
    # test_numba_main()
    c, _ = numba_main_2(test, 1000, np.array([0]), 0)
    test_numba_main_2()
    test_normal_main_2()


if __name__ == '__main__':
    main()

