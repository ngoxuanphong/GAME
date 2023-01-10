from CheckEnv import check_env
from CheckPlayer import check_agent
from CheckEnvAgent import CreateFolder, load_module_player, add_game_to_syspath, setup_game, save_json
from AutoTrain import train_agent_by_level

from setup import SHOT_PATH, time_run_game, N_AGENT, N_GAME, PASS_LEVEL
import pandas as pd
import importlib.util
import numpy as np
import sys, time, shutil
import json, os, itertools

env_name = 'Splendor_v2'
list_name_agent = ['TimeBasedAlgorithm', 'An', 'BiasAlgorihm', 'MultiDimensionAlgorithm', 'StateBasedAlgorithm', 'Trang']
add_game_to_syspath(env_name)
__env__ = setup_game(env_name)
dict_level = {}
combinations = []
dict_result = {}
for agent in list_name_agent:
    list_name_agent_copy = list_name_agent.copy()
    list_name_agent_copy.remove(agent)
    dict_level[env_name] = {}
    p0 = load_module_player(agent).Test
    combis = enumerate(itertools.combinations(list_name_agent_copy, __env__.getAgentSize()-1))

    level_max = 1
    for level0, combination in combis:
        list_level = [0, 0, list(combination)]
        dict_level[env_name][str(level0+1)] = list_level
        level_max += 1
    
    save_json(f'{SHOT_PATH}Log/level_game{env_name}.json', dict_level)
    per_player_here = list(np.load(f'{SHOT_PATH}Agent/{agent}/Data/{env_name}_0/Train.npy',allow_pickle=True))
    win_total_player = 0
    for level in range(level_max):
        win, per = __env__.numba_main_2(p0, 100, per_player_here, level)
        win_total_player += win
        print(level, win)
    print(agent, win_total_player)
    dict_result[agent] = win_total_player
print(dict_result)