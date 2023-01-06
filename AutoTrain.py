from setup import time_run_game, SHOT_PATH
from CreateLog import logger

from getFromServer import get_notifi_server, state_train_server
import sys, os, time, json
import pandas as pd
import numpy as np
import importlib.util
from CheckEnvAgent import CreateFolder, update_json, setup_game, load_module_player, add_game_to_syspath
import warnings 
warnings.filterwarnings('ignore')
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning,NumbaExperimentalFeatureWarning, NumbaWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)
warnings.simplefilter('ignore', category=NumbaWarning)


# df_run = pd.read_json(f'{SHOT_PATH}Log/State.json')
# df_agent = pd.read_json(f'{SHOT_PATH}Log/StateAgent.json')
# df_env = pd.read_json(f'{SHOT_PATH}Log/StateEnv.json')

# list_tong = list(df_run.index)
# dict_agent = json.load(open(f'{SHOT_PATH}Log/agent_all.json'))
# dict_level = json.load(open(f'{SHOT_PATH}Log/level_game.json'))
PASS_LEVEL = 1000
COUNT_PLAYER_TRAIN_NEW_ENV = 10


def train_agent_by_level(game, path_save_player, level, _p1_, time_loop, *args): #arg = 1 là train theo mốc, k phải train hệ thống mới
    PerDataStartTrain = _p1_.DataAgent()
    if time_loop == 0: time_loop = 1
    win = 0
    start_train = time.time()
    for _time_ in range(time_loop):
        win, PerDataStartTrain = game.numba_main_2(_p1_.Train, 10000, PerDataStartTrain, level)
        np.save(f'{path_save_player}Train.npy',PerDataStartTrain)
        end_train = time.time()
        if (end_train - start_train) > time_run_game:
            win, PerDataStartTrain = game.numba_main_2(_p1_.Test, 10000, PerDataStartTrain, level)
            break
    return win

def check_code(game, player):
    _p1_ = load_module_player(player)
    win, PerDataStartTrain = game.numba_main_2(_p1_.Train, 100, _p1_.DataAgent(), 0)
    
    start_10000 = time.time()
    win, PerDataStartTrain = game.numba_main_2(_p1_.Train, 10000, _p1_.DataAgent(), 0)
    end_10000 = time.time()
    return True, int(time_run_game/(end_10000 - start_10000))

def save_json(path_save_json, dict_save):
    with open(path_save_json, 'w') as f:
        json.dump(dict_save, f, indent=1)

def read_edit_save_df_level(df_run, col, id, win):
    df_run = pd.read_json(f'{SHOT_PATH}Log/State.json')
    df_run[col][id] = win
    df_run.to_json(f'{SHOT_PATH}Log/State.json')

def train_new_env():
    df_run = pd.read_json(f'{SHOT_PATH}Log/State.json')
    dict_level = json.load(open(f'{SHOT_PATH}Log/level_game.json'))
    for col in df_run.columns[7:]:
            env_name = col
        # if dict_level[env_name]['Can_Split_Level'] == 'False': #6/1/2023
            for id in df_run.index:
                if pd.isna(df_run[col][id]):
                    agent_name = df_run['ID'][id]
                    read_edit_save_df_level(df_run, col, id, 'RUNNING')
                    add_game_to_syspath(env_name)
                    
                    level = 0
                    path_save_player = CreateFolder(agent_name, env_name, level)
                    __env__ = setup_game(env_name)
                    _p1_ = load_module_player(agent_name)

                    check_agent, time_loop = check_code(__env__, agent_name)
                    if check_agent == True:
                        win = train_agent_by_level(__env__, path_save_player, level, _p1_, time_loop*10000)
                        read_edit_save_df_level(df_run, col, id, win)
                    else:
                        read_edit_save_df_level(df_run, col, id, 0)
                    break
            else:
                continue
            break

    # for col in df_run.columns[7:]: #6/1/2023
    #     if pd.isna(df_run[col]).any() == False:
    #         dict_level = json.load(open(f'{SHOT_PATH}Log/level_game.json'))
    #         dict_level[col]['Can_Split_Level'] = 'True'
    #         save_json(f'{SHOT_PATH}Log/level_game.json', dict_level)

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

def after_train(current_lv, env_name, agent_name, win, condition_pass_level):
    dict_level = json.load(open(f'{SHOT_PATH}Log/level_game.json'))
    dict_agent = json.load(open(f'{SHOT_PATH}Log/agent_all.json'))

    if win > condition_pass_level:
        dict_agent[agent_name][env_name]['State_level'][2] = current_lv
    else:
        dict_agent[agent_name][env_name]['State_level'][2] = current_lv - 1

    state_train_server(agent_name, False, current_lv)

    if current_lv == dict_level[env_name]['level_max']:
        dict_agent[agent_name][env_name]['State_level'][0] = -1

    elif current_lv < dict_level[env_name]['level_max'] - 2:
        dict_agent[agent_name][env_name]['State_level'][0] = -1
        dict_agent[agent_name]['Agent Save'] = False

    else:
        if win <= condition_pass_level:
            if current_lv == dict_level[env_name]['level_max'] -2:
                dict_agent[agent_name][env_name]['State_level'][0] = 0
                dict_agent[agent_name]['Agent Save'] = False
            else:
                dict_agent[agent_name][env_name]['State_level'][0] = -1
        else:
            __env__ = setup_game(env_name)
            data_agent_env = list(np.load(f'{SHOT_PATH}Agent/{agent_name}/Data/{env_name}_{current_lv}/Train.npy',allow_pickle=True))
            add_game_to_syspath(env_name)
            p0 = load_module_player(agent_name).Test
            check_new_level = False

            for level_test in range(dict_level[env_name]['level_max'], current_lv, -1):
                if len(dict_agent[agent_name][env_name][str(level_test)]) == 0:
                    win_test, per = __env__.numba_main_2(p0, 1000, data_agent_env, level_test)
                    
                    print(level_test, win_test*10)
                    dict_agent = json.load(open(f'{SHOT_PATH}Log/agent_all.json'))
                    if win_test*10 > condition_pass_level:
                        dict_agent[agent_name][env_name]['State_level'][1] = level_test
                        check_new_level = True
                        if str(level_test) not in dict_agent[agent_name][env_name]:
                            dict_agent[agent_name][env_name][level_test] = []
                        break
            if check_new_level == False:
                dict_agent[agent_name][env_name]['State_level'][1] = current_lv + 1
            dict_agent[agent_name][env_name]['State_level'][0] = 0


    save_json(f'{SHOT_PATH}Log/level_game.json', dict_level)
    save_json(f'{SHOT_PATH}Log/agent_all.json', dict_agent)

def choose_game_level_train(agent_name):
    dict_level = json.load(open(f'{SHOT_PATH}Log/level_game.json'))
    dict_agent = json.load(open(f'{SHOT_PATH}Log/agent_all.json'))

    for env_name in dict_agent[agent_name]:
        if env_name not in ['Elo', 'First train', 'Level Train', 'Agent Save']:
            state_train_env = dict_agent[agent_name][env_name]['State_level']

            if state_train_env[0] == 0:
                if dict_agent[agent_name]['Agent Save'] == True: #Nếu là agent đang mạnh thì train tiếp level 
                    if state_train_env[1] < dict_level[env_name]['level_max'] - 2:
                        level_train = dict_level[env_name]['level_max'] - 2
                    else:
                        level_train = state_train_env[1]
                        if level_train < 1: level_train = 1
                        if level_train > dict_level[env_name]['level_max']: level_train = 0
                           
                else:
                    level_train = dict_level[env_name]['level_max']//2

                if len(dict_agent[agent_name][env_name][str(level_train)]) == 0:
                    dict_agent[agent_name][env_name]['State_level'][0] = 1
                    save_json(f'{SHOT_PATH}Log/agent_all.json', dict_agent)

                    state_train_server(agent_name, env_name, level_train)
                    return env_name, level_train

    for env_name in dict_agent[agent_name]:
        if env_name not in ['Elo', 'First train', 'Level Train', 'Agent Save']:
            state_train_env = dict_agent[agent_name][env_name]['State_level']
            if state_train_env[0] == 0 and state_train_env[1] != 0:#Game này đang train game này và level này đã qua level 5
                if dict_agent[agent_name]['Agent Save'] == True:
                    level_train = state_train_env[1]

                    dict_agent[agent_name][env_name]['State_level'][0] = 1
                    save_json(f'{SHOT_PATH}Log/agent_all.json', dict_agent)

                    state_train_server(agent_name, env_name, level_train)
                    return env_name, level_train
    return False, False

def add_to_strong_agent(env_name, agent_name, level, win, condition_pass_level):
    dict_level = json.load(open(f'{SHOT_PATH}Log/level_game.json'))
    dict_agent = json.load(open(f'{SHOT_PATH}Log/agent_all.json'))
    if level == dict_level[env_name]['level_max'] and win > condition_pass_level: # (10000/(__env__.getAgentSize()))
        check_dk = True
        for env_name_check in dict_agent[agent_name]:
            if env_name_check not in ['Elo', 'First train', 'Level Train', 'Agent Save']:
                check_condition_1_env = False

                level_max = dict_level[env_name_check]['level_max']
                for id_level in range(level_max, level_max-3, -1):
                    if str(id_level) in dict_agent[agent_name][env_name_check]:
                        if len(dict_agent[agent_name][env_name_check][str(id_level)]) > 0:
                            if dict_agent[agent_name][env_name_check][str(id_level)][0] > condition_pass_level:
                                check_condition_1_env = True
                                break

                if check_condition_1_env == False:
                    print('false tai day', env_name_check, agent_name)
                    check_dk = False
            
        df_run = pd.read_json(f'{SHOT_PATH}Log/State.json')
        if check_dk == True: 
            if agent_name not in list(df_run["ID"]):
                if os.path.exists(f'{SHOT_PATH}Agent/{agent_name}/Data/{env_name}_0/') == False:
                    folder_name_old = f'{SHOT_PATH}Agent/{agent_name}/Data/{env_name}_{level}/'
                    folder_name_new = f'{SHOT_PATH}Agent/{agent_name}/Data/{env_name}_0/'
                    os.rename(folder_name_old, folder_name_new)

                id_add_agent = len(df_run)
                df_run.loc[id_add_agent] = 0
                df_run['ID'][id_add_agent] = agent_name
                df_run[env_name][id_add_agent] = win//10
                df_run.to_json(f'{SHOT_PATH}Log/State.json')

def train_agent():
    dict_agent = json.load(open(f'{SHOT_PATH}Log/agent_all.json'))
    dict_level = json.load(open(f'{SHOT_PATH}Log/level_game.json'))
    for agent_name in dict_agent:
        env_name, level = choose_game_level_train(agent_name)
        if env_name != False:
            add_game_to_syspath(env_name)
            path_save_player = CreateFolder(agent_name, env_name, level)
            __env__ = setup_game(env_name)
            _p1_ = load_module_player(agent_name)
            
            check_agent, time_loop = check_code(__env__, agent_name)
            if check_agent == True:
                start = time.time()
                win = train_agent_by_level(__env__, path_save_player, level, _p1_, time_loop*1000, 1)
                end = time.time()
                
                condition_pass_level = 10000/__env__.getAgentSize()
                add_to_strong_agent(env_name, agent_name, level, win, condition_pass_level)

                after_train(level, env_name, agent_name, win, condition_pass_level)

                dict_agent = json.load(open(f'{SHOT_PATH}Log/agent_all.json'))

                dict_agent[agent_name][env_name][str(level)] = [win, end - start]
                save_json(f'{SHOT_PATH}Log/agent_all.json', dict_agent)
                break


def run_autotrain():
    # test_and_add_new_level()
    train_new_env()
    # train_agent()

# run_autotrain()

# state_train_server('system0_1672838310', False, 2)