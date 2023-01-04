import os
import shutil
text_add = """
import numpy as np
from numba import njit
import sys, os
from setup import SHOT_PATH
import importlib.util
game_name = sys.argv[1]

def setup_game(game_name):
    spec = importlib.util.spec_from_file_location('env', f"{SHOT_PATH}base/{game_name}/env.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module 
    spec.loader.exec_module(module)
    return module

env = setup_game(game_name)
# """
# shutil.move('Agent_player.py', 'Agent_player.txt')
# with open('Agent_player.txt', 'r') as original: data = original.read()
# with open('Agent_player.txt', 'w') as modified: modified.write(f"{text_add}\n" + data)
# shutil.move('Agent_player.txt', 'Agent_player.py')

from CreateLog import logger
logger.info('Hi')