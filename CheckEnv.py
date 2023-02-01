import numpy as np
from numba import njit, jit
from system import logger
import functools
import multiprocessing.pool

import warnings 
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning,NumbaExperimentalFeatureWarning, NumbaWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)
warnings.simplefilter('ignore', category=NumbaWarning)

COUNT_TEST = 1000

def timeout(max_timeout):
    """Timeout decorator, parameter in seconds."""
    def timeout_decorator(item):
        """Wrap the original function."""
        @functools.wraps(item)
        def func_wrapper(*args, **kwargs):
            """Closure for function."""
            pool = multiprocessing.pool.ThreadPool(processes=1)
            async_result = pool.apply_async(item, args, kwargs)
            # raises a TimeoutError if execution exceeds max_timeout
            return async_result.get(max_timeout)
        return func_wrapper
    return timeout_decorator

def CheckAllFunc(_env_, BOOL_CHECK_ENV, msg):
    for func in ['getActionSize','getStateSize','getAgentSize','getReward','getValidActions','normal_main','numba_main_2']:
        try:
            getattr(_env_, func)
        except:
            msg.append(f'Không có hàm: {func}')
            BOOL_CHECK_ENV = False
    return BOOL_CHECK_ENV


def CheckReturn(_env_, BOOL_CHECK_ENV, msg):
    for func in ['getActionSize','getStateSize','getAgentSize']:
        try:
            func_ = getattr(_env_, func)
            out = func_()
            if type(out) != int and type(out) != np.int64:
                msg.append(f'ham {func} tra sai dau ra: dau ra yeu cau int, dau ra hien tai: {type(out)}')
                BOOL_CHECK_ENV = False
        except:
            pass
    return BOOL_CHECK_ENV


# @timeout(1000)
def RunGame(_env_, BOOL_CHECK_ENV, msg):
    @njit()
    def test_numba(p_state, per_file):
        arr_action = _env_.getValidActions(p_state)

        if _env_.getReward(p_state) != -1:
            per_file[0] += 1
        if _env_.getReward(p_state) == 1:
            per_file[1] += 1
        if np.min(p_state) < 0:
            per_file[2] = 1
        if len(p_state) != _env_.getStateSize():
            per_file[3] = 1
        if len(arr_action) != _env_.getActionSize():
            per_file[4] = 1

        arr_action = np.where(arr_action == 1)[0]
        act_idx = np.random.randint(0, len(arr_action))
        return arr_action[act_idx], per_file

    def test_no_numba(p_state, per_file):
        arr_action = _env_.getValidActions(p_state)
        if p_state.dtype != np.float64:
            per_file[5] = 1
        if arr_action.dtype != np.float64:
            per_file[6] = 1
        arr_action = np.where(arr_action == 1)[0]
        act_idx = np.random.randint(0, len(arr_action))
        return arr_action[act_idx], per_file

    try:
        per = np.array([0, 0, 0, 0, 0, 0, 0]) #end, win end, state âm, state thay đổi, actions thay đổi, type state, type action
        win, per = _env_.numba_main_2(test_numba, COUNT_TEST, per, 0)

        if type(win) != int and type(win) != np.int64:
            msg.append('hàm numba_main_2 trả ra sai đầu ra, yêu cầu int')
            BOOL_CHECK_ENV = False
        if per[0] != COUNT_TEST:
            msg.append(f'numba_main_2, Số trận kết thúc khác với số trận test, {per[0]}')
            BOOL_CHECK_ENV = False
        if per[1] != win:
            msg.append(f'numba_main_2, Số trận thắng khi kết thúc khác với sô trận check. Thắng khi dùng getReward: {per[1]}, win: {win}')
            BOOL_CHECK_ENV = False
        if per[2] == 1:
            msg.append(f'State đang bị âm')
            BOOL_CHECK_ENV = False
        if per[3] == 1:
            msg.append(f'len STATE đang thay đổi trong quá trình chạy game')
            BOOL_CHECK_ENV = False
        if per[4] == 1:
            msg.append(f'len ACTION đang thay đổi trong quá trình chạy game')
            BOOL_CHECK_ENV = False

        try:
            per = np.array([0, 0, 0, 0, 0, 0, 0])
            win, per = _env_.numba_main_2(test_no_numba, COUNT_TEST, per, 0)
            if per[5] == 1:
                msg.append(f'STATE đang trả ra sai output')
                BOOL_CHECK_ENV = False
            if per[6] == 1:
                msg.append(f'array ACTION đang trả ra sai output')
                BOOL_CHECK_ENV = False  
        except:
            msg.append('hàm numba_main_2 không train được với agent không numba, cần đổi n_game_numba với one_game_numba từ njit() thành jit(). Mẹo đổi tất cả @njit thành @njit()')
            BOOL_CHECK_ENV = False
        
    except:
        msg.append(f'hàm numba_main_2 đang bị lỗi')
        BOOL_CHECK_ENV = False

    try:
        per = [0, 0, 0, 0, 0, 0, 0]
        win, per = _env_.normal_main([test_numba]*_env_.getAgentSize(), COUNT_TEST, per)
        if type(win) != list and type(win) != np.ndarray:
            msg.append('hàm normal_main trả ra sai đầu ra')
            BOOL_CHECK_ENV = False
        if per[0] != COUNT_TEST*_env_.getAgentSize():
            msg.append(f'Normal_main, Số trận kết thúc khác với số trận test, {per[0]}')
            BOOL_CHECK_ENV = False

    except:
        msg.append(f'hàm normal_main đang bị lỗi')
        BOOL_CHECK_ENV = False

    return BOOL_CHECK_ENV

def CheckRunGame(_env_, BOOL_CHECK_ENV, msg):
    # try:
        BOOL_CHECK_ENV = RunGame(_env_, BOOL_CHECK_ENV, msg)
    # except:
    #     msg.append('Khả năng là bị vòng lặp vô hạn')
    #     BOOL_CHECK_ENV = False
        return BOOL_CHECK_ENV



def check_env(_env_):
    BOOL_CHECK_ENV = True
    msg = []
    BOOL_CHECK_ENV = CheckAllFunc(_env_, BOOL_CHECK_ENV, msg)
    BOOL_CHECK_ENV = CheckReturn(_env_, BOOL_CHECK_ENV, msg)
    BOOL_CHECK_ENV = CheckRunGame(_env_, BOOL_CHECK_ENV, msg)
    return BOOL_CHECK_ENV, msg