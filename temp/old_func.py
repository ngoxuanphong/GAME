from setup import time_run_game, SHOT_PATH
from CreateLog import logger

from getFromServer import get_notifi_server, state_train_server
import sys, os, time, json
import pandas as pd
import numpy as np
import importlib.util
from CheckEnvAgent import CreateFolder, update_json, setup_game, load_module_player, add_game_to_syspath, save_json
import warnings 
warnings.filterwarnings('ignore')
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning,NumbaExperimentalFeatureWarning, NumbaWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)
warnings.simplefilter('ignore', category=NumbaWarning)


def test_and_add_new_level():
    df_run = pd.read_json(f'{SHOT_PATH}Log/State.json')
    dict_level = json.load(open(f'{SHOT_PATH}Log/level_game.json'))

    for col in df_run.columns[7:]:
        env_name = col
        if dict_level[env_name]['Can_Split_Level'] == 'True':

            dict_level[env_name]['Can_Split_Level'] = 'Testing'
            save_json(f'{SHOT_PATH}Log/level_game.json', dict_level)

            __env__ = setup_game(env_name)
            bool_add_new_level = get_lv(__env__.getAgentSize()-1, env_name)
            dict_level = json.load(open(f'{SHOT_PATH}Log/level_game.json'))
            list_tong = list(df_run.index)
            if bool_add_new_level: #Đã thêm các level mới rồi thì test lại các agent còn lại với level vừa thêm
                id_con_lai = [i for i in list_tong if i not in dict_level[env_name]['id_remove']]
                for id_train in id_con_lai:
                    agent_name = df_run['ID'][id_train]
                    
                    level_max = dict_level[env_name]['level_max']
                    data_agent_env = list(np.load(f'{SHOT_PATH}Agent/{agent_name}/Data/{env_name}_0/Train.npy',allow_pickle=True))

                    add_game_to_syspath(env_name)
                    p0 = load_module_player(agent_name).Test
                    win, per = __env__.numba_main_2(p0, 1000, data_agent_env, level_max)

                    df_run = pd.read_json(f'{SHOT_PATH}Log/State.json')
                    df_run[col][id_train] = win
                    df_run.to_json(f'{SHOT_PATH}Log/State.json')

            dict_level = json.load(open(f'{SHOT_PATH}Log/level_game.json'))
            dict_level[env_name]['Can_Split_Level'] = 'True'
            save_json(f'{SHOT_PATH}Log/level_game.json', dict_level)
            break

def get_lv(count_agent, env_name):
    df_run = pd.read_json(f'{SHOT_PATH}Log/State.json')
    # dict_level = json.load(open(f'{SHOT_PATH}Log/level_game.json'))
    df_run_con_lai = df_run[df_run[env_name] != 'DONE']
    khong_pass_level = list(df_run_con_lai[df_run_con_lai[env_name] < PASS_LEVEL/(count_agent+1)].index) #PASS_LEVEL/(count_agent+1)

    dict_level = json.load(open(f'{SHOT_PATH}Log/level_game.json'))
    if len(khong_pass_level) > 0:
        df_run[env_name][khong_pass_level] = 'DONE'
        df_run_con_lai = df_run[df_run[env_name] != 'DONE']
        for id_have_id_list in khong_pass_level:
            if id_have_id_list not in dict_level[env_name]['id_remove']:
                dict_level[env_name]['id_remove'].append(id_have_id_list)
        save_json(f'{SHOT_PATH}Log/level_game.json', dict_level)
        df_run.to_json(f'{SHOT_PATH}Log/State.json')

    df_run_con_lai = df_run_con_lai[df_run_con_lai[env_name] > PASS_LEVEL/(count_agent+1)] #PASS_LEVEL/(count_agent+1)
    infor_level = df_run_con_lai.sort_values(by = env_name)
    bool_add_new_level = False

    if len(infor_level) >= count_agent:
        all_id_lv = list(infor_level[:count_agent].index)
        win_count = list(infor_level[:count_agent][env_name])
        name_agents = list(infor_level[:count_agent]['ID'])

        for id_have_id_list in all_id_lv:
            if id_have_id_list not in dict_level[env_name]['id_remove']:
                dict_level[env_name]['id_remove'].append(id_have_id_list)

        dict_level[env_name]['level_max'] += 1
        lv_max = dict_level[env_name]['level_max']

        for agent_name in name_agents:
            folder_name_old = f'{SHOT_PATH}Agent/{agent_name}/Data/{env_name}_0/'
            folder_name_new = f'{SHOT_PATH}Agent/{agent_name}/Data/{env_name}_{lv_max}/'
            if os.path.exists(folder_name_old) == True and os.path.exists(folder_name_new) == False:
                os.rename(folder_name_old, folder_name_new)
            if os.path.exists(folder_name_new):
                pass
            if os.path.exists(folder_name_old) == False and os.path.exists(folder_name_new) == False:
                logger.debug(f'Split Level:  Agent_name: {agent_name}, Env_name: {env_name}, Level_add: {lv_max}, Folder old: {folder_name_old}:{os.path.exists(folder_name_old)}, Folder new: {folder_name_new}:{os.path.exists(folder_name_new)}')
                folder_have = os.listdir(f'{SHOT_PATH}Agent/{agent_name}/Data/')
                if len(folder_have) > 0:
                    folder_name_change = f'{SHOT_PATH}Agent/{agent_name}/Data/{folder_have[-1]}/'
                    os.rename(folder_name_change, folder_name_new)

        
        if lv_max not in dict_level[env_name]:
            dict_level[env_name][lv_max] = [all_id_lv, win_count, name_agents]

        df_run = pd.read_json(f'{SHOT_PATH}Log/State.json')
        df_run[env_name][all_id_lv] = 'DONE'
        bool_add_new_level = 'True'

        df_run.to_json(f'{SHOT_PATH}Log/State.json')
        save_json(f'{SHOT_PATH}Log/level_game.json', dict_level)
    return bool_add_new_level
