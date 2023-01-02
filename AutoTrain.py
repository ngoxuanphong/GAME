from setup import game_name, time_run_game, path, SHOT_PATH
from getFromServer import get_notifi_server
import sys, os, time, json
import pandas as pd
import numpy as np
import importlib.util

import warnings 
warnings.filterwarnings('ignore')
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning,NumbaExperimentalFeatureWarning, NumbaWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)
warnings.simplefilter('ignore', category=NumbaWarning)


df_run = pd.read_json(f'{SHOT_PATH}Log/State.json')
df_agent = pd.read_json(f'{SHOT_PATH}Log/StateAgent.json')
df_env = pd.read_json(f'{SHOT_PATH}Log/StateEnv.json')

list_tong = list(df_run.index)
dict_agent = json.load(open(f'{SHOT_PATH}Log/agent_all.json'))
dict_level = json.load(open(f'{SHOT_PATH}Log/level_game.json'))
PASS_LEVEL = 250
COUNT_PLAYER_TRAIN_NEW_ENV = 10

def setup_game(game_name):
    spec = importlib.util.spec_from_file_location('env', f"{SHOT_PATH}base/{game_name}/env.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module 
    spec.loader.exec_module(module)
    return module
 

def load_module_player(player):
    return  importlib.util.spec_from_file_location('Agent_player', f"{SHOT_PATH}Agent/{player}/Agent_player.py").loader.load_module()


def CreateFolder(player, game_name, level): #Tên folder của người chơi
    path_data = f'{SHOT_PATH}Agent/{player}/Data'
    if not os.path.exists(path_data):
        os.mkdir(path_data)
    path_save_player = f'{SHOT_PATH}Agent/{player}/Data/{game_name}_{level}/'
    if not os.path.exists(path_save_player):
        os.mkdir(path_save_player)
    return path_save_player


def test_data(game, _p1_, path_save_player, level):
    PerDataStartTrain = list(np.load(f'{path_save_player}Train.npy',allow_pickle=True))
    win, per = game.numba_main_2(_p1_.Agent, 1000, PerDataStartTrain, level)
    return win/10


def train(game, path_save_player, level, _p1_, time_loop, *args): #arg = 1 là train theo mốc, k phải train hệ thống mới
    PerDataStartTrain = _p1_.DataAgent()
    for time in range(time_loop):
        win, PerDataStartTrain = game.numba_main_2(_p1_.Agent, 10000, PerDataStartTrain, level)
        np.save(f'{path_save_player}Train.npy',PerDataStartTrain)

        if (len(args) > 0) and (win > 1.1*10000/game.getAgentSize()):
            # print('win', win, 'time loop', time)
            return win 
    np.save(f'{path_save_player}Train.npy',PerDataStartTrain)
    return win

def check_code(game, player):
    _p1_ = load_module_player(player)
    win, PerDataStartTrain = game.numba_main_2(_p1_.Agent, 100, _p1_.DataAgent(), 0)
    
    start_10000 = time.time()
    win, PerDataStartTrain = game.numba_main_2(_p1_.Agent, 10000, _p1_.DataAgent(), 0)
    end_10000 = time.time()
    return True, int(time_run_game/(end_10000 - start_10000))

def save_json(path_save_json, dict_save):
    with open(path_save_json, 'w') as f:
        json.dump(dict_save, f)

def train_new_env(df_run, dict_level):
    for col in df_run.columns[7:]:
        env_name = col
        print(dict_level[env_name]['Can_Split_Level'])
        if dict_level[env_name]['Can_Split_Level'] == 'False':
            for id in df_run.index[:COUNT_PLAYER_TRAIN_NEW_ENV]:
                if pd.isna(df_run[col][id]):
                    agent_name = df_run['ID'][id]
                    df_run[col][id] = 'RUNNING'
                    df_run.to_json(f'{SHOT_PATH}Log/State.json')

                    if len(sys.argv) >= 2:
                        sys.argv = [sys.argv[0]]
                    sys.argv.append(env_name)
                    
                    level = 0
                    print('Tạo folder', env_name, agent_name, 'level', level)
                    path_save_player = CreateFolder(agent_name, env_name, level)
                    env = setup_game(env_name)
                    _p1_ = load_module_player(agent_name)

                    check_agent, time_loop = check_code(env, agent_name)
                    print('Check Agent, time_loop', check_agent, time_loop)
                    if check_agent == True:
                        print('train', env_name, path_save_player, level, agent_name, time_loop)
                        win = train(env, path_save_player, level, _p1_, time_loop)
                        print('Tỉ lệ thắng sau khi train', win)
                        df_run[col][id] = win
                        df_run.to_json(f'{SHOT_PATH}Log/State.json')
                    else:
                        df_run[col][id] = 0

        dict_level[env_name]['Can_Split_Level'] = 'True'
        save_json('Log/level_game.json', dict_level)

def get_lv(count_agent, df_run, env_name, dict_level):
    df_run_con_lai = df_run[df_run[env_name] != 'DONE']
    khong_pass_level = list(df_run_con_lai[df_run_con_lai[env_name] < PASS_LEVEL].index)

    if len(khong_pass_level) > 0:
        # print('khong_pass_level', khong_pass_level, df_run['ID'][khong_pass_level])
        df_run[env_name][khong_pass_level] = 'DONE'
        df_run_con_lai = df_run[df_run[env_name] != 'DONE']

    df_run.to_json(f'{SHOT_PATH}Log/State.json')

    df_run_con_lai = df_run_con_lai[df_run_con_lai[env_name] > PASS_LEVEL]
    infor_level = df_run_con_lai.sort_values(by = env_name)
    bool_add_new_level = False

    # print(infor_level)
    if len(infor_level) >= count_agent:
        all_id_lv = list(infor_level[:count_agent].index)
        win_count = list(infor_level[:count_agent][env_name])
        name_agents = list(infor_level[:count_agent]['ID'])

        # print('Agent pass', name_agents)
        for id_have_id_list in all_id_lv:
            if id_have_id_list not in dict_level[env_name]['id_remove']:
                dict_level[env_name]['id_remove'].append(id_have_id_list)

        dict_level[env_name]['level_max'] += 1
        lv_max = dict_level[env_name]['level_max']
        # print(lv_max)

        for agent_name in name_agents:
            folder_name_old = f'{SHOT_PATH}Agent/{agent_name}/Data/{env_name}_0/'
            folder_name_new = f'{SHOT_PATH}Agent/{agent_name}/Data/{env_name}_{lv_max}/'
            os.rename(folder_name_old, folder_name_new)
        
        if lv_max not in dict_level[env_name]:
            dict_level[env_name][lv_max] = [all_id_lv, win_count, name_agents]

        df_run[env_name][all_id_lv] = 'DONE'
        bool_add_new_level = 'True'

        df_run.to_json(f'{SHOT_PATH}Log/State.json')
        save_json('Log/level_game.json', dict_level)
    return df_run, dict_level, bool_add_new_level


def test_and_add_new_level(df_run, dict_level):
    for col in df_run.columns[7:]:
        env_name = col
        if dict_level[env_name]['Can_Split_Level'] == 'True':
            for id_have_id_list in list(df_run[env_name].loc[COUNT_PLAYER_TRAIN_NEW_ENV:].index):
                if id_have_id_list not in dict_level[env_name]['id_remove']:
                    dict_level[env_name]['id_remove'].append(id_have_id_list)

            for id in df_run.index[:COUNT_PLAYER_TRAIN_NEW_ENV]:
                if df_run[col][id] != 'DONE':
                    env = setup_game(env_name)
                    df_run, dict_level, bool_add_new_level = get_lv(env.getAgentSize()-1, df_run, env_name, dict_level)
                    
                    if bool_add_new_level: #Đã thêm các level mới rồi thì test lại các agent còn lại với level vừa thêm
                        id_con_lai = [i for i in list_tong if i not in dict_level[env_name]['id_remove']]
                        for id_train in id_con_lai:
                            agent_name = df_run['ID'][id_train]
                            
                            level_max = dict_level[env_name]['level_max']
                            data_agent_env = list(np.load(f'{SHOT_PATH}Agent/{agent_name}/Data/{env_name}_0/Train.npy',allow_pickle=True))

                            if len(sys.argv) >= 2:
                                sys.argv = [sys.argv[0]]
                            sys.argv.append(env_name)
                            p0 = load_module_player(agent_name).Agent

                            win, per = env.numba_main_2(p0, 1000, data_agent_env, level_max)
                            # print('Train agent:', agent_name, 'win', win)
                            df_run[col][id_train] = win
                            df_run.to_json(f'{SHOT_PATH}Log/State.json')



def choose_game_level_train(dict_agent, dict_level, agent_name):
    level_train = int((dict_agent[agent_name]['Elo'] - 1200)/10)
    if dict_agent[agent_name]['Elo'] <= 1180:
        return False, False
        
    for env_name in dict_agent[agent_name]:
        if env_name != 'Elo':
            if dict_level[env_name]['level_max'] < level_train:
                level_train = dict_level[env_name]['level_max']

            if str(level_train) in dict_agent[agent_name][env_name]:
                if len(dict_agent[agent_name][env_name][str(level_train)]) == 0:
                    dict_agent[agent_name][env_name][str(level_train)] = [0, 0]
                    return env_name, level_train

    return False, False

def train_agent(dict_agent, dict_level):
    for agent_name in dict_agent:
        env_name, level = choose_game_level_train(dict_agent, dict_level, agent_name)
        if env_name != False:
            if len(sys.argv) >= 2:
                sys.argv = [sys.argv[0]]
            sys.argv.append(env_name)

            path_save_player = CreateFolder(agent_name, env_name, level)
            env = setup_game(env_name)
            _p1_ = load_module_player(agent_name)
            
            check_agent, time_loop = check_code(env, agent_name)
            if level < 3:
                time_run = time_run_game//(3-level)
                time_loop = time_loop//(3-level)

            if check_agent == True:
                start = time.time()
                win = train(env, path_save_player, level, _p1_, time_loop, 1)
                end = time.time()
                if (win > 1.1*10000/env.getAgentSize()):
                    elo_add = win/(1.1*10000/env.getAgentSize()) + (100*(level+1))/((end-start))
                    dict_agent[agent_name]['Elo'] += elo_add
                else:
                    dict_agent[agent_name]['Elo'] -= 0.5**level*50

                dict_agent[agent_name][env_name][str(level)] = [win, end - start]
                # print('Elo', dict_agent[agent_name]['Elo'])

                save_json(f'{SHOT_PATH}Log/agent_all.json', dict_agent)
                get_notifi_server('Agent', 'TRAINING', agent_name, dict_agent[agent_name]['Elo'])

while True:
    train_new_env(df_run, dict_level)
    test_and_add_new_level(df_run, dict_level)
    train_agent(dict_agent, dict_level)