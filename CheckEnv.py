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
    for func in ['getActionSize','getStateSize','getAgentSize','getReward','getValidActions','normal_main','numba_main','numba_main_2']:
        try:
            getattr(_env_, func)
        except:
            logger.warn(f'Không có hàm: {func}')
            msg.append(f'Không có hàm: {func}')
            BOOL_CHECK_ENV = False
    return BOOL_CHECK_ENV


def CheckReturn(_env_, BOOL_CHECK_ENV, msg):
    for func in ['getActionSize','getStateSize','getAgentSize']:
        try:
            func_ = getattr(_env_, func)
            out = func_()
            if type(out) != int and type(out) != np.int64:
                logger.warn(f'ham {func} tra sai dau ra: dau ra yeu cau int, dau ra hien tai: {type(out)}')
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
        arr_action = np.where(arr_action == 1)[0]
        act_idx = np.random.randint(0, len(arr_action))
        if _env_.getReward(p_state) != -1:
            per_file[0] += 1
        if _env_.getReward(p_state) == 1:
            per_file[1] += 1
        return arr_action[act_idx], per_file

    def test_no_numba(p_state, per_file):
        arr_action = _env_.getValidActions(p_state)
        arr_action = np.where(arr_action == 1)[0]
        act_idx = np.random.randint(0, len(arr_action))
        return arr_action[act_idx], per_file

    try:
        per = np.array([0, 0])
        win, per = _env_.numba_main_2(test_numba, COUNT_TEST, per, 0)
        if type(win) != int and type(win) != np.int64:
            logger.warn('hàm numba_main_2 trả ra sai đầu ra')
            msg.append('hàm numba_main_2 trả ra sai đầu ra')
            BOOL_CHECK_ENV = False
        if per[0] != COUNT_TEST:
            logger.warn(f'Số trận kết thúc khác với số trận test, {per[0]}')
            msg.append(f'Số trận kết thúc khác với số trận test, {per[0]}')
            BOOL_CHECK_ENV = False
        if per[1] != win:
            logger.warn(f'Số trận thắng khi kết thúc khác với sô trận check. Thắng khi dùng getReward: {per[1]}, win: {win}')
            msg.append(f'Số trận thắng khi kết thúc khác với sô trận check. Thắng khi dùng getReward: {per[1]}, win: {win}')
            BOOL_CHECK_ENV = False
        try:
            per = np.array([0])
            win, per = _env_.numba_main_2(test_no_numba, COUNT_TEST, per, 0)
        except:
            logger.warn('hàm numba_main_2 không train được với agent không numba, cần đổi n_game_numba với one_game_numba từ njit() thành jit')
            msg.append('hàm numba_main_2 không train được với agent không numba, cần đổi n_game_numba với one_game_numba từ njit() thành jit')
            BOOL_CHECK_ENV = False
    except:
        logger.warn(f'hàm numba_main_2 đang bị lỗi')
        msg.append(f'hàm numba_main_2 đang bị lỗi')
        BOOL_CHECK_ENV = False

    try:
        per = [0, 0]
        win, per = _env_.normal_main([test_numba]*_env_.getAgentSize(), COUNT_TEST, per)
        if type(win) != list:
            logger.warn('hàm normal_main trả ra sai đầu ra')
            msg.append('hàm normal_main trả ra sai đầu ra')
            BOOL_CHECK_ENV = False
        if per[0] != COUNT_TEST*_env_.getAgentSize():
            logger.warn(f'Số trận kết thúc khác với số trận test, {per[0]}')
            msg.append(f'Số trận kết thúc khác với số trận test, {per[0]}')
            BOOL_CHECK_ENV = False

    except:
        logger.warn(f'hàm normal_main đang bị lỗi')
        msg.append(f'hàm normal_main đang bị lỗi')
        BOOL_CHECK_ENV = False

        
    return BOOL_CHECK_ENV

def CheckRunGame(_env_, BOOL_CHECK_ENV, msg):
    try:
        BOOL_CHECK_ENV = RunGame(_env_, BOOL_CHECK_ENV, msg)
    except:
        logger.warn('Khả năng là bị vòng lặp vô hạn')
        msg.append('Khả năng là bị vòng lặp vô hạn')
        BOOL_CHECK_ENV = False
    return BOOL_CHECK_ENV



def check_env(_env_):
    BOOL_CHECK_ENV = True
    msg = []
    BOOL_CHECK_ENV = CheckAllFunc(_env_, BOOL_CHECK_ENV, msg)
    BOOL_CHECK_ENV = CheckReturn(_env_, BOOL_CHECK_ENV, msg)
    BOOL_CHECK_ENV = CheckRunGame(_env_, BOOL_CHECK_ENV, msg)
    # BOOL_CHECK_ENV = check_lv1(_env_, BOOL_CHECK_ENV, msg)
    return BOOL_CHECK_ENV, msg