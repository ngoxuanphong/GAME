import sys
import time
from CheckEnv import check_env
from CheckPlayer import check_agent
from server.mysql_connector import get_db_cursor
from getFromServer import get_notifi_server, update_notificate_by_id, copy_new_agent, copy_new_env
from numba import njit

import pandas as pd
import importlib.util
import numpy as np
import json, os
from setup import SHOT_PATH
from CreateLog import logger

def setup_game(game_name):
    spec = importlib.util.spec_from_file_location('env', f"{SHOT_PATH}base/{game_name}/env.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module 
    spec.loader.exec_module(module)
    return module
    
def load_module_player(player):
    spec = importlib.util.spec_from_file_location('Agent_player', f"{SHOT_PATH}Agent/{player}/Agent_player.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module 
    spec.loader.exec_module(module)
    return module

def add_game_to_syspath(env_name):
    if len(sys.argv) >= 2:
        sys.argv = [sys.argv[0]]
    sys.argv.append(env_name)

def save_json(path_save_json, dict_save):
    with open(path_save_json, 'w') as f:
        json.dump(dict_save, f, indent=1)

def CreateFolder(player, game_name, level): #Tên folder của người chơi
    path_data = f'{SHOT_PATH}Agent/{player}/Data'
    if not os.path.exists(path_data):
        os.mkdir(path_data)
    path_save_player = f'{SHOT_PATH}Agent/{player}/Data/{game_name}_{level}/'
    if not os.path.exists(path_save_player):
        os.mkdir(path_save_player)
    return path_save_player

def update_json():
    dict_level = json.load(open(f'{SHOT_PATH}Log/level_game.json'))
    dict_agent = json.load(open(f'{SHOT_PATH}Log/agent_all.json'))

    for agent in dict_agent:
        for env_name in dict_level:
            if env_name not in dict_agent[agent]:
                dict_agent[agent][env_name] = {'State_level':[0, 0 , 0]}
            for level in range(dict_level[env_name]['level_max']+1):
                if str(level) not in dict_agent[agent][env_name]:
                    dict_agent[agent][env_name][level] = []
    save_json(f'{SHOT_PATH}Log/agent_all.json', dict_agent)


def fix_player():
    df_agent = pd.read_json(f'{SHOT_PATH}Log/StateAgent.json')

    for id in df_agent.index:
        state_agent = df_agent.loc[id, 'CHECK']
        if pd.isna(state_agent):

                agent_name = df_agent['AGENT'][id]
                df_agent.loc[id, 'CHECK'] = 'CHECKING'
                df_agent.to_json(f'{SHOT_PATH}Log/StateAgent.json')

                get_notifi_server('Agent', 'CHECKING', agent_name)
                print(agent_name)
            # try:
                agent_test = importlib.util.spec_from_file_location('Agent_player', f"A:\AutoTrain\GAME\Agent\{agent_name}\Agent_player.py").loader.load_module()
                bool_check_agent, msg = check_agent(agent_test)
                df_agent = pd.read_json(f'{SHOT_PATH}Log/StateAgent.json')
                if bool_check_agent == True: #Sửa lại file excel trạng thái
                    df_agent.loc[id, 'CHECK'] = 'DONE'

                    dict_agent = json.load(open(f'{SHOT_PATH}Log/agent_all.json'))
                    dict_agent[agent_name] = {'Elo':1200,
                                            'First train': False, 
                                            'Level Train': 0, 
                                            'Agent Save': True}
                    get_notifi_server('Agent', 'NOBUG', agent_name)
                    save_json(f'{SHOT_PATH}Log/agent_all.json', dict_agent)
                else:
                    df_agent.loc[id, 'CHECK'] = 'BUG1'
                    df_agent.loc[id, 'NOTE'] = str(msg)

                    get_notifi_server('Agent', 'BUG', agent_name)
                    
                df_agent.to_json(f'{SHOT_PATH}Log/StateAgent.json')
            # except:
            #     df_agent.loc[id, 'CHECK'] = 'BUG2'
            #     df_agent.to_json(f'{SHOT_PATH}Log/StateAgent.json')
            #     get_notifi_server('Agent', 'BUG', agent_name)

                break

def sql_get_id_by_systen_name(env_name):
    mycursor, mydb = get_db_cursor()
    sql = f''' SELECT hs.ID, su.Name
                    FROM  HistorySystem hs
                    Left join auth_user au on au.id = hs.UserID
                    Left join SystemUser su on su.SystemID = hs.SystemID
                    Left join Notificate n on n.NotificateID = hs.NotificateID
                    Where su.Name = %s'''
    val = [env_name]
    mycursor.execute(sql, val)
    data_env = mycursor.fetchall()
    if len(data_env) > 0:
        return data_env[0][0]

def read_edit_save_df_env(id, col, msg):
    df_env = pd.read_json(f'{SHOT_PATH}Log/StateEnv.json')
    df_env.loc[id, col] = msg
    df_env.to_json(f'{SHOT_PATH}Log/StateEnv.json')


def check_env_level_1(env_name_test):
    @njit()
    def test_numba(p_state, per_file):
        arr_action = env_test.getValidActions(p_state)
        arr_action = np.where(arr_action == 1)[0]
        act_idx = np.random.randint(0, len(arr_action))
        return arr_action[act_idx], per_file
    agent_name_test = 'train_to_check_level_env'

    if os.path.exists(f'{SHOT_PATH}Log/check_system_about_level.json') == False:
        save_json(f'{SHOT_PATH}Log/check_system_about_level.json', {})

    dict_env_test_level = json.load(open(f'{SHOT_PATH}Log/check_system_about_level.json'))
    if env_name_test not in dict_env_test_level:
        dict_env_test_level[env_name_test] = {'1': [0, 0, [agent_name_test, agent_name_test, agent_name_test]]}
    
    save_json(f'{SHOT_PATH}Log/check_system_about_level.json', dict_env_test_level)
    add_game_to_syspath(env_name_test)
    env_test = setup_game(env_name_test)
    agent_test = load_module_player(agent_name_test)

    path_save_test = CreateFolder(agent_name_test, env_name_test, 1)
    perSaveToChecking = agent_test.DataAgent()
    np.save(f'{path_save_test}Train.npy',perSaveToChecking)

    time.sleep(2)
    try:
        win , per = env_test.numba_main_2(test_numba, 1000, np.array([0]), 1, 1)
        print(win)
        return True
    except:
        return False

def fix_env():
    df_env = pd.read_json(f'{SHOT_PATH}Log/StateEnv.json')

    for id in df_env.index:
        state_env = df_env.loc[id, 'CHECK']
        if pd.isna(state_env):
            env_name = df_env['ENV'][id]
            logger.info(f'Check env{env_name}')
            read_edit_save_df_env(id, 'CHECK', 'CHECKING')

            ID_env = sql_get_id_by_systen_name(env_name)
            update_notificate_by_id(ID_env, 'CHECKING')

            try:
                env = importlib.util.spec_from_file_location('env', f"{SHOT_PATH}base/{env_name}/env.py").loader.load_module()
                bool_check_env, msg = check_env(env)
                if bool_check_env == True:
                    bool_check_level = check_env_level_1(env_name)
                    if bool_check_level == True:
                        read_edit_save_df_env(id, 'CHECK', 'DONE')
                        update_notificate_by_id(ID_env, 'NO BUG')

                        df_run = pd.read_json(f'{SHOT_PATH}Log/State.json')
                        df_run[env_name] = np.nan
                        df_run.to_json(f'{SHOT_PATH}Log/State.json')
                    else:
                        update_notificate_by_id(ID_env, 'BUGLV1')
                        read_edit_save_df_env(id, 'CHECK', 'BUG')
                        read_edit_save_df_env(id, 'NOTE', 'Bug level 1')
                else:
                    update_notificate_by_id(ID_env, 'BUG')
                    read_edit_save_df_env(id, 'CHECK', 'BUG')
                    read_edit_save_df_env(id, 'NOTE', str(msg))

                dict_level = json.load(open(f'{SHOT_PATH}Log/level_game.json'))
                dict_level_all = json.load(open(f'{SHOT_PATH}Log/level_game_all.json'))
                state_train_agent = [0 for i in range(len(dict_level_all["1"]["Agents Name"]))]
                score_train_agent = [0 for i in range(len(dict_level_all["1"]["Agents Name"]))]
                agent_name_train = dict_level_all["1"]["Agents Name"]
                dict_level[env_name] = {"Can_Split_Level": 'False',
                                        "level_max": 0,
                                        "1": [state_train_agent, score_train_agent, agent_name_train]}
            except:
                update_notificate_by_id(ID_env, 'BUG')
                read_edit_save_df_env(id, 'CHECK', 'BUG')
                read_edit_save_df_env(id, 'NOTE', str(msg))
            
            save_json(f'{SHOT_PATH}Log/level_game.json', dict_level)
            update_json()
            break

def __checking_all__():
    if os.path.exists(f'{SHOT_PATH}Log/agent_all.json') == False:
        save_json(f'{SHOT_PATH}Log/agent_all.json', {})

    if os.path.exists(f'{SHOT_PATH}Log/level_game.json') == False:
        save_json(f'{SHOT_PATH}Log/level_game.json', {})

    copy_new_agent()
    copy_new_env()

    update_json()
    fix_player()
    fix_env()
    update_json()

__checking_all__()




