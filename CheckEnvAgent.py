from CheckEnv import check_env
from CheckPlayer import check_agent
from server.mysql_connector import mydb, mycursor
from getFromServer import get_notifi_server

from setup import path
import pandas as pd
import importlib.util
import numpy as np
import json, os
from setup import SHOT_PATH


def save_json(path_save_json, dict_save):
    with open(path_save_json, 'w') as f:
        json.dump(dict_save, f)

def update_json(dict_agent, dict_level):
    for agent in dict_agent:
        for env_name in dict_level:
            if env_name not in dict_agent[agent]:
                dict_agent[agent][env_name] = {}
            for level in range(dict_level[env_name]['level_max']+1):
                if str(level) not in dict_agent[agent][env_name]:
                    dict_agent[agent][env_name][level] = []
    save_json(f'{SHOT_PATH}Log/agent_all.json', dict_agent)


def fix_player(df_run, df_agent, dict_agent, dict_level):
    for id in df_agent.index:
        state_agent = df_agent.loc[id, 'CHECK']
        if pd.isna(state_agent):

            agent_name = df_agent['AGENT'][id]
            df_agent.loc[id, 'CHECK'] = 'CHECKING'
            df_agent.to_json(f'{SHOT_PATH}Log/StateAgent.json')

            get_notifi_server('Agent', 'CHECKING', agent_name)

            agent_test = importlib.util.spec_from_file_location('Agent_player', f"A:\AutoTrain\GAME\Agent\{agent_name}\Agent_player.py").loader.load_module()
            bool_check_agent, msg = check_agent(agent_test)

            if bool_check_agent == True: #Sửa lại file excel trạng thái
                df_agent.loc[id, 'CHECK'] = 'DONE'
                id_add_agent = len(df_run)
                df_run.loc[id_add_agent] = np.nan
                df_run['ID'][id_add_agent] = agent_name
                df_run.to_json(f'{SHOT_PATH}Log/State.json')
                dict_agent[agent_name] = {'Elo':1200}

                get_notifi_server('Agent', 'NOBUG', agent_name)
            else:
                df_agent.loc[id, 'CHECK'] = 'BUG'
                df_agent.loc[id, 'NOTE'] = str(msg)

                get_notifi_server('Agent', 'CHECKING', agent_name)
                
            df_agent.to_json(f'{SHOT_PATH}Log/StateAgent.json')
            update_json(dict_agent, dict_level)
            save_json(f'{SHOT_PATH}Log/agent_all.json', dict_agent)


def fix_env(df_run, df_env, dict_agent, dict_level):
    for id in df_env.index:
        state_env = df_env.loc[id, 'CHECK']
        if pd.isna(state_env):
            env_name = df_env['ENV'][id]
            df_env.loc[id, 'CHECK'] = 'CHECKING'
            df_env.to_json(f'{SHOT_PATH}Log/StateEnv.json')
            env = importlib.util.spec_from_file_location('env', f"A:/GAME/base/{env_name}/env.py").loader.load_module()
            bool_check_env, msg = check_env(env)
            if bool_check_env == True:
                df_env.loc[id, 'CHECK'] = 'DONE'
                df_run[env_name] = np.nan
                df_run.to_json(f'{SHOT_PATH}Log/State.json')
            else:
                df_env.loc[id, 'CHECK'] = 'BUG'
                df_env.loc[id, 'NOTE'] = str(msg)

            df_env.to_json(f'{SHOT_PATH}Log/StateEnv.json')

            dict_level[env_name] = {"Can_Split_Level": 'False',
                                    "level_max": 0,
                                    "id_remove":[]}
            update_json(dict_agent, dict_level)
            save_json(f'{SHOT_PATH}Log/level_game.json', dict_level)


def __checking_all__():
    df_run = pd.read_json(f'{SHOT_PATH}Log/State.json')
    df_agent = pd.read_json(f'{SHOT_PATH}Log/StateAgent.json')
    df_env = pd.read_json(f'{SHOT_PATH}Log/StateEnv.json')

    if os.path.exists(f'{SHOT_PATH}Log/agent_all.json') == False:
        save_json(f'{SHOT_PATH}Log/agent_all.json', {})

    if os.path.exists(f'{SHOT_PATH}Log/agent_all.json') == False:
        save_json(f'{SHOT_PATH}Log/agent_all.json', {})

    dict_level = json.load(open(f'{SHOT_PATH}Log/level_game.json'))
    dict_agent = json.load(open(f'{SHOT_PATH}Log/agent_all.json'))

    update_json(dict_agent, dict_level)
    fix_player(df_run, df_agent, dict_agent, dict_level)
    # fix_env(df_run, df_env, dict_agent, dict_level)

__checking_all__()

