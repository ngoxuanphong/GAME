import shutil
from setup import time_run_game, SHOT_PATH, PASS_LEVEL, COUNT_TRAIN, COUNT_TEST, N_AGENT
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


def train_agent_by_level(game, path_save_player, level, _p1_, time_loop, *args): #arg = 1 là train theo mốc, k phải train hệ thống mới
    PerDataStartTrain = _p1_.DataAgent()
    if time_loop == 0: time_loop = 1
    win = 0
    start_train = time.time()
    for _time_ in range(time_loop):
        win, PerDataStartTrain = game.numba_main_2(_p1_.Train, COUNT_TRAIN, PerDataStartTrain, level)
        np.save(f'{path_save_player}Train.npy',PerDataStartTrain)
        end_train = time.time()
        if (end_train - start_train) > time_run_game:
            win, PerDataStartTrain = game.numba_main_2(_p1_.Test, COUNT_TEST, PerDataStartTrain, level)
            break
    return win

def check_code(game, player):
    _p1_ = load_module_player(player)
    win, PerDataStartTrain = game.numba_main_2(_p1_.Train, 100, _p1_.DataAgent(), 0)
    return True, 1000000


def read_edit_save_df_level(df_run, col, id, win):
    df_run = pd.read_json(f'{SHOT_PATH}Log/State.json')
    df_run[col][id] = win
    df_run.to_json(f'{SHOT_PATH}Log/State.json')

# def train_new_env():
#     df_run = pd.read_json(f'{SHOT_PATH}Log/State.json')
#     dict_level = json.load(open(f'{SHOT_PATH}Log/level_game.json'))
#     for id in df_run.index:
#         for col in df_run.columns[7:]:
#                 env_name = col
#         # if dict_level[env_name]['Can_Split_Level'] == 'False': #6/1/2023
#                 if pd.isna(df_run[col][id]):
#                     agent_name = df_run['ID'][id]
#                     read_edit_save_df_level(df_run, col, id, 'RUNNING')
#                     add_game_to_syspath(env_name)
                    
#                     level = 0
#                     path_save_player = CreateFolder(agent_name, env_name, level)
#                     __env__ = setup_game(env_name)
#                     _p1_ = load_module_player(agent_name)

#                     check_agent, time_loop = check_code(__env__, agent_name)
#                     if check_agent == True:
#                         win = train_agent_by_level(__env__, path_save_player, level, _p1_, time_loop*COUNT_TRAIN)
#                         read_edit_save_df_level(df_run, col, id, win)
#                     else:
#                         read_edit_save_df_level(df_run, col, id, 0)
#                     break
#         else:
#             continue
#         break

    # for col in df_run.columns[7:]: #6/1/2023
    #     if pd.isna(df_run[col]).any() == False:
    #         dict_level = json.load(open(f'{SHOT_PATH}Log/level_game.json'))
    #         dict_level[col]['Can_Split_Level'] = 'True'
    #         save_json(f'{SHOT_PATH}Log/level_game.json', dict_level)



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
        if current_lv == 1:
            if win <= condition_pass_level: #Sua 7/1/2023
                dict_agent[agent_name]['Agent Save'] = False
        else:
            dict_level[env_name][str(current_lv)][6][1] += 1
            __env__ = setup_game(env_name)
            if dict_level[env_name][str(current_lv)][6][1] >= 0.8*(__env__.getAgentSize()-1):
                dict_level[env_name][str(current_lv)][6][0] = 1

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
            __p0__ = load_module_player(agent_name).Test
            check_new_level = False

            for level_test in range(dict_level[env_name]['level_max'], current_lv, -1):
                if len(dict_agent[agent_name][env_name][str(level_test)]) == 0:
                    win_test, per = __env__.numba_main_2(__p0__, 1000, data_agent_env, level_test)
                    
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
                           
                else: #Agent yếu chỉ train 1 nửa ròi trả ra kết quả
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
    if win > condition_pass_level: # (COUNT_TRAIN/(__env__.getAgentSize()))
        check_dk = True
        for env_name_check in dict_agent[agent_name]:
            if env_name_check not in ['Elo', 'First train', 'Level Train', 'Agent Save']:
                if dict_agent[agent_name][env_name_check]['State_level'][0] != -1:
                    check_dk = False

        if check_dk == True and dict_agent[agent_name]['Agent Save'] == True:
            dict_level_all = json.load(open(f'{SHOT_PATH}Log/level_game_all.json'))
            LMAX = dict_level_all['Level max']
            if dict_level_all[str(LMAX)]["Can Train"] == 'False':
                dict_level_all[str(LMAX)]["Agents Name"].append(agent_name)
                save_json(f'{SHOT_PATH}Log/level_game_all.json', dict_level_all)

def train_agent():
    dict_agent = json.load(open(f'{SHOT_PATH}Log/agent_all.json'))
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
                win = train_agent_by_level(__env__, path_save_player, level, _p1_, time_loop*100000, 1)
                end = time.time()
                
                # condition_pass_level = COUNT_TEST/__env__.getAgentSize()
                condition_pass_level = 0 #DECHECK

                after_train(level, env_name, agent_name, win, condition_pass_level)

                add_to_strong_agent(env_name, agent_name, level, win, condition_pass_level)
                #Cần sửa lại
                
                dict_agent = json.load(open(f'{SHOT_PATH}Log/agent_all.json'))
                dict_agent[agent_name][env_name][str(level)] = [win, end - start]
                save_json(f'{SHOT_PATH}Log/agent_all.json', dict_agent)
                break


def get_level(): #Hàm này để check xem đã có đủ agent để lập một level mới hay chưa
    dict_level_all = json.load(open(f'{SHOT_PATH}Log/level_game_all.json'))
    level = dict_level_all['Level max']

    check_get_new_level = True
    dict_level = json.load(open(f'{SHOT_PATH}Log/level_game.json'))
    if level != 1:
        for env_name in dict_level:
            if str(level) in dict_level[env_name]:
                if dict_level[env_name][str(level)][6][0] == 0:
                    check_get_new_level = False
            else: check_get_new_level = False

    if str(level) in dict_level_all:
        if (len(dict_level_all[str(level)]['Agents Name']) >= N_AGENT)and (check_get_new_level == True): #Thêm level mới khi có level cũ đã đủ số lượng agent
            dict_level_all[str(level)]['Can Train'] = "True"
            dict_level_all['Level max'] += 1
            dict_level_all[level+1] = {'Can Train': "False",
                                        'Agents Name': []}
        save_json(f'{SHOT_PATH}Log/level_game_all.json', dict_level_all)

def CopyAgent(agent_old, agent_new): #Tên folder của người chơi
    path_data = f'{SHOT_PATH}Agent/{agent_new}/'
    if not os.path.exists(path_data):
        os.mkdir(path_data)
    time.sleep(0.5)
    shutil.copy2(f'{SHOT_PATH}Agent/{agent_old}/Agent_player.py', f'{SHOT_PATH}Agent/{agent_new}/Agent_player.py')

#List agent ở trong một game thì xếp thứ tự theo rank
def check_copy_agent_need(env_name, level): #Khi không đủ agent để làm lv1 thì lấy thêm một agent train lại và làm level của game đó
    dict_level = json.load(open(f'{SHOT_PATH}Log/level_game.json'))
    add_game_to_syspath(env_name)
    __env__  = setup_game(env_name)
    current_agent_level = dict_level[env_name][str(level)][2]
    if (__env__.getAgentSize()-1) > len(current_agent_level) and len(current_agent_level) > 0: #Nếu số agent cần nhiều hơn số agent đang có của game này
        count_agent_need = (__env__.getAgentSize()-1) - len(current_agent_level)
        for id_need in range(count_agent_need):
            if id_need >= len(current_agent_level):
                id_need = 0
            id_copy = 0
            while True:
                name_agent_copy = f'{current_agent_level[id_need]}{id_copy}'
                if name_agent_copy not in dict_level[env_name][str(level)][2]:
                    dict_level[env_name][str(level)][2].append(name_agent_copy)
                    dict_level[env_name][str(level)][0].append(0)
                    dict_level[env_name][str(level)][1].append(0)

                    CopyAgent(current_agent_level[id_need], name_agent_copy)
                    break
                id_copy += 1
        save_json(f'{SHOT_PATH}Log/level_game.json', dict_level)

def train_copy_agent(env_name, level, type_train): #1 là đã train, 0 là chưa train, -1 là đang train
    dict_level = json.load(open(f'{SHOT_PATH}Log/level_game.json'))
    dict_agent = json.load(open(f'{SHOT_PATH}Log/agent_all.json'))
    level_train = level - 1
    add_game_to_syspath(env_name)
    __env__  = setup_game(env_name)
    # condition_pass_level = COUNT_TEST/(__env__.getAgentSize())

    condition_pass_level = 0 #DECHECK

    if type_train == 'Train Copy':
        id_state_save = 0 
        id_win_score_save = 1
        id_agent_name_save = 2
    if type_train == 'Train New':
        id_state_save = 3
        id_win_score_save = 4
        id_agent_name_save = 5
    current_agent_level = dict_level[env_name][str(level)][id_agent_name_save]
    state_agent_level = dict_level[env_name][str(level)][id_state_save]
    for id_agent, state_agent in enumerate(state_agent_level):
        if state_agent == 0: #Chưa train
            agent_name = current_agent_level[id_agent]
            check_trained = False
            if agent_name in dict_agent:
                if str(level_train) in dict_agent[agent_name][env_name]:
                    if len(dict_agent[agent_name][env_name][str(level_train)]) > 0:
                        if dict_agent[agent_name][env_name][str(level_train)][0] > 0:
                            check_trained = True

            if check_trained == False:
                __p0__ = load_module_player(agent_name)
                dict_level[env_name][str(level)][id_state_save][id_agent] = -1 #Đang train
                save_json(f'{SHOT_PATH}Log/level_game.json', dict_level)

                path_save_agent_copy = CreateFolder(agent_name, env_name, level)
                print('train_copy_agent', __env__, __p0__, __p0__.Train)
                win = train_agent_by_level(__env__, path_save_agent_copy, level_train, __p0__, 100000000000000)

                dict_level = json.load(open(f'{SHOT_PATH}Log/level_game.json'))
                dict_level[env_name][str(level)][id_state_save][id_agent] = 1 #Đã train xong
                dict_level[env_name][str(level)][id_win_score_save][id_agent] = win
                save_json(f'{SHOT_PATH}Log/level_game.json', dict_level)
            
            else:
                dict_level[env_name][str(level)][id_state_save][id_agent] = 1
                dict_level[env_name][str(level)][id_win_score_save][id_agent] = dict_agent[agent_name][env_name][str(level_train)][0]
                save_json(f'{SHOT_PATH}Log/level_game.json', dict_level)
            break


def get_new_level_for_env(): #Thêm level mới cho hệ thống khi đã đủ điều kiện
    dict_level = json.load(open(f'{SHOT_PATH}Log/level_game.json'))
    dict_level_all = json.load(open(f'{SHOT_PATH}Log/level_game_all.json'))

    for env_name in dict_level:
        LMAX_env = dict_level[env_name]["level_max"]+1

        if str(LMAX_env) in dict_level[env_name]:
            if len(dict_level[env_name][str(LMAX_env)][0]) > 0:
                print(dict_level[env_name]["level_max"])
                if min(dict_level[env_name][str(LMAX_env)][0]) == 1: #Khi đã đủ điều kiện lập level mới
                    __env__ = setup_game(env_name)
                    if (__env__.getAgentSize() - 1) <= len(dict_level[env_name][str(LMAX_env)][0]):
                        dict_level[env_name]["level_max"] = LMAX_env
                        save_json(f'{SHOT_PATH}Log/level_game.json', dict_level)

        dict_level = json.load(open(f'{SHOT_PATH}Log/level_game.json'))
        LMAX_env = dict_level[env_name]["level_max"]
        LMAX_env_new = LMAX_env + 1

        if str(LMAX_env) in dict_level[env_name]:
            if min(dict_level[env_name][str(LMAX_env)][0]) == 1:
                if LMAX_env_new <= dict_level_all["Level max"] and (str(LMAX_env_new) not in dict_level[env_name]): 
                    if dict_level_all[str(LMAX_env_new)]['Can Train'] == 'True': #Khi lưu trữ đã đủ cho level này
                        __env__ = setup_game(env_name)
                        if (__env__.getAgentSize() - 1) <= len(dict_level[env_name][str(LMAX_env)][0]):
                            lst_agent_of_level = dict_level_all[str(LMAX_env_new)]['Agents Name']
                            lst_score_agent_of_level = [0 for i in range(N_AGENT)]
                            lst_state_agent_of_level = [0 for i in range(N_AGENT)]
                            dict_level[env_name][str(LMAX_env_new)] = [[],[],[],lst_state_agent_of_level,lst_score_agent_of_level,lst_agent_of_level, [0, 0]]
                            save_json(f'{SHOT_PATH}Log/level_game.json', dict_level)
            
def train_all_agent_of_level(): #Train test các agent của level đó
    dict_level_all = json.load(open(f'{SHOT_PATH}Log/level_game_all.json'))
    dict_level = json.load(open(f'{SHOT_PATH}Log/level_game.json'))
    for env_name in dict_level:
        LMAX_env = dict_level[env_name]["level_max"]+1
        if str(LMAX_env) in dict_level[env_name]: #Có 2 loại, train N_agent để chọn ra top agent và train, copy các agent thiếu
            if LMAX_env == 1:
                type_train = 'Train Copy'
                check_copy_agent_need(env_name, LMAX_env)
                train_copy_agent(env_name, LMAX_env, type_train)
                break
            else:
                if min(dict_level[env_name][str(LMAX_env)][3]) == 1: # dùng copy agent thiếu
                    type_train = 'Train Copy'
                    check_copy_agent_need(env_name, LMAX_env)
                    train_copy_agent(env_name, LMAX_env, type_train)
                else: #Train agent lưu trữ, chọn ra agent đủ điều kiện
                    type_train = 'Train New'
                    train_copy_agent(env_name, LMAX_env, type_train)
                    dict_level = json.load(open(f'{SHOT_PATH}Log/level_game.json'))
                    if min(dict_level[env_name][str(LMAX_env)][3]) == 1:
                        lst_agent_of_level = np.array(dict_level[env_name][str(LMAX_env)][5])
                        lst_score_agent_of_level = np.array(dict_level[env_name][str(LMAX_env)][4])

                        __env__ = setup_game(env_name)
                        # condition_pass_level = COUNT_TEST/(__env__.getAgentSize())

                        condition_pass_level = 0 #DECHECK
                        id_agent_pass_level, = np.where(lst_score_agent_of_level > condition_pass_level)

                        if len(id_agent_pass_level) > 0:
                            id_agent_pass_level_sort = id_agent_pass_level[np.argsort(lst_score_agent_of_level[id_agent_pass_level])][::-1]
                            
                            dict_level[env_name][str(LMAX_env)][2] = list(lst_agent_of_level[id_agent_pass_level_sort])
                            dict_level[env_name][str(LMAX_env)][1] = [int(i) for i in lst_score_agent_of_level[id_agent_pass_level_sort]]
                            dict_level[env_name][str(LMAX_env)][0] = [1 for i in range(len(id_agent_pass_level))] #Gán thành đã train

                            save_json(f'{SHOT_PATH}Log/level_game.json', dict_level)
                break

# def run_autotrain():
    # test_and_add_new_level()
    # train_new_env()
    # train_agent()

# run_autotrain()

train_agent()
get_level()
get_new_level_for_env()
train_all_agent_of_level()
