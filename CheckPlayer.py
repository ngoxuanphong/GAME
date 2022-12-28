import sys

from  system import logger
from base.Splendor_v3 import env

if len(sys.argv) >= 2:
    sys.argv = [sys.argv[0]]
sys.argv.append('Splendor_v3')

COUNT_TEST = 10000

def CheckAllFunc(Agent, BOOL_CHECK_ENV, msg):
    for func in ['DataAgent', 'Agent']:
        try:
            getattr(Agent, func)
        except:
            logger.warn(f'Không có hàm: {func}')
            msg.append(f'Không có hàm: {func}')
            BOOL_CHECK_ENV = False
    return BOOL_CHECK_ENV, msg

def CheckRunGame(Agent, BOOL_CHECK_ENV, msg):
    try:
        per = Agent.DataAgent()
        win, per = env.numba_main_2(Agent.Agent, COUNT_TEST, per, 0)
        # win, per = env.normal_main_2(Agent.Agent, COUNT_TEST, per, 0)
    except:
        logger.warn(f'Agent đang bị lỗi')
        msg.append(f'Agent đang bị lỗi')
        BOOL_CHECK_ENV = False
    return BOOL_CHECK_ENV, msg


def check_agent(Agent):

    BOOL_CHECK_ENV = True
    msg = []

    BOOL_CHECK_ENV, msg = CheckAllFunc(Agent, BOOL_CHECK_ENV, msg)
    BOOL_CHECK_ENV, msg = CheckRunGame(Agent, BOOL_CHECK_ENV, msg)
    return BOOL_CHECK_ENV, msg