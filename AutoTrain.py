from setup import game_name, time_run_game, path, SHOT_PATH

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


df_run = pd.read_excel('A:\State.xlsx')
df_agent = pd.read_excel('A:\StateAgent.xlsx')
df_env = pd.read_excel('A:\StateEnv.xlsx')

list_tong = list(df_run.index)
dict_level = json.load(open(f'{SHOT_PATH}level_game.json'))
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


def CreateFolder(player, game_name, time_run_game): #Tên folder của người chơi
    path_data = f'{SHOT_PATH}Agent/{player}/Data'
    if not os.path.exists(path_data):
        os.mkdir(path_data)
    path_save_player = f'{SHOT_PATH}Agent/{player}/Data/{game_name}_{time_run_game}/'
    if not os.path.exists(path_save_player):
        os.mkdir(path_save_player)
    return path_save_player


def test_data(game, p1, path_save_player, level):
    PerDataStartTrain = list(np.load(f'{path_save_player}Train.npy',allow_pickle=True))
    win, per = game.numba_main_2(p1.Agent, 1000, PerDataStartTrain, level)
    return win/10


def train(game, path_save_player, level, p1, time_loop):
    PerDataStartTrain = p1.DataAgent()
    for time in range(time_loop):
        win, PerDataStartTrain = game.numba_main_2(p1.Agent, 10000, PerDataStartTrain, level)
        np.save(f'{path_save_player}Train.npy',PerDataStartTrain)
    np.save(f'{path_save_player}Train.npy',PerDataStartTrain)
    return win

def check_code(game, player):
    p1 = load_module_player(player)
    win, PerDataStartTrain = game.numba_main_2(p1.Agent, 100, p1.DataAgent(), 0)
    
    start_10000 = time.time()
    win, PerDataStartTrain = game.numba_main_2(p1.Agent, 10000, p1.DataAgent(), 0)
    end_10000 = time.time()
    return True, int(time_run_game/(end_10000 - start_10000))

def save_json(path_save_json, dict_save):
    with open(path_save_json, 'w') as f:
        json.dump(dict_save, f, indent= 4)

def train_new_env(df_run, dict_level):
    for col in df_run.columns[7:]:
        env_name = col
        print(dict_level[env_name]['Can_Split_Level'])
        if dict_level[env_name]['Can_Split_Level'] == 'False':
            for id in df_run.index[:COUNT_PLAYER_TRAIN_NEW_ENV]:
                if pd.isna(df_run[col][id]):
                    agent_name = df_run['ID'][id]
                    df_run[col][id] = 'RUNNING'
                    df_run.to_excel('A:\State.xlsx', index=False)

                    if len(sys.argv) >= 2:
                        sys.argv = [sys.argv[0]]
                    sys.argv.append(env_name)
                    
                    print('Tạo folder', env_name, agent_name, time_run_game)
                    path_save_player = CreateFolder(agent_name, env_name, time_run_game)
                    env = setup_game(env_name)
                    p1 = load_module_player(agent_name)

                    check_agent, time_loop = check_code(env, agent_name)
                    print('Check Agent, time_loop', check_agent, time_loop)
                    if check_agent == True:
                        print('train', env, path_save_player, 0, p1, time_loop)
                        win = train(env, path_save_player, 0, p1, time_loop)
                        print('Tỉ lệ thắng sau khi train', win)
                        df_run[col][id] = win
                        df_run.to_excel('A:\State.xlsx', index=False)

        dict_level[env_name]['Can_Split_Level'] = 'True'
        save_json('level_game.json', dict_level)

def get_lv(count_agent, df_run, env_name, dict_level):
    df_run_con_lai = df_run[df_run[env_name] != 'DONE']
    khong_pass_level = list(df_run_con_lai[df_run_con_lai[env_name] < PASS_LEVEL].index)

    if len(khong_pass_level) > 0:
        print('khong_pass_level', khong_pass_level, df_run['ID'][khong_pass_level])
        df_run[env_name][khong_pass_level] = 'DONE'

    df_run.to_excel('A:\State.xlsx', index=False)

    df_run_con_lai = df_run_con_lai[df_run_con_lai[env_name] > PASS_LEVEL]
    infor_level = df_run_con_lai.sort_values(by = env_name)
    bool_add_new_level = False

    if len(infor_level) >= count_agent:
        all_id_lv = list(infor_level[:count_agent].index)
        win_count = list(infor_level[:count_agent][env_name])
        name_agents = list(infor_level[:count_agent]['ID'])

        print('Agent pass', name_agents)
        dict_level[env_name]['id_remove'] += all_id_lv
        dict_level[env_name]['level_max'] += 1
        lv_max = dict_level[env_name]['level_max']
        if lv_max not in dict_level[env_name]:
            dict_level[env_name][lv_max] = [all_id_lv, win_count, name_agents]

        df_run[env_name][all_id_lv] = 'DONE'
        bool_add_new_level = 'True'

        df_run.to_excel('A:\State.xlsx', index=False)
        save_json('level_game.json', dict_level)
    return df_run, dict_level, bool_add_new_level


def test_and_add_new_level(df_run, dict_level):
    for col in df_run.columns[7:]:
        env_name = col
        if dict_level[env_name]['Can_Split_Level'] == 'True':
            # df_run[env_name].loc[COUNT_PLAYER_TRAIN_NEW_ENV:] = 'DONE'
            dict_level[env_name]['id_remove'] += list(df_run[env_name].loc[COUNT_PLAYER_TRAIN_NEW_ENV:].index)

            df_run.to_excel('A:\State.xlsx', index=False)
            for id in df_run.index[:COUNT_PLAYER_TRAIN_NEW_ENV]:
                if df_run[col][id] != 'DONE':
                    env = setup_game(env_name)
                    df_run, dict_level, bool_add_new_level = get_lv(env.getAgentSize()-1, df_run, env_name, dict_level)
                    
                    if bool_add_new_level: #Đã thêm các level mới rồi thì test lại các agent còn lại với level vừa thêm
                        id_con_lai = [i for i in list_tong if i not in dict_level[env_name]['id_remove']]
                        for id_train in id_con_lai:
                            agent_name = df_run['ID'][id_train]
                            data_agent_env = list(np.load(f'{SHOT_PATH}Agent/{agent_name}/Data/{env_name}_{time_run_game}/Train.npy',allow_pickle=True))
                            
                            level_max = dict_level[env_name]['level_max']

                            if len(sys.argv) >= 2:
                                sys.argv = [sys.argv[0]]
                            sys.argv.append(env_name)
                            p0 = load_module_player(agent_name).Agent

                            win, per = env.numba_main_2(p0, 1000, data_agent_env, level_max)
                            print('Train agent:', agent_name, 'win', win)
                            df_run[col][id_train] = win
                            df_run.to_excel('A:\State.xlsx', index=False)

train_new_env(df_run, dict_level)
test_and_add_new_level(df_run, dict_level)
