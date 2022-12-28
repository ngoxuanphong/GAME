from CheckEnv import check_env
from CheckPlayer import check_agent
from setup import path
import pandas as pd
import importlib.util
import numpy as np
import json, os
from setup import SHOT_PATH

df_run = pd.read_excel('A:\State.xlsx')
df_agent = pd.read_excel('A:\StateAgent.xlsx')
df_env = pd.read_excel('A:\StateEnv.xlsx')

def save_json(path_save_json, dict_save):
    with open(path_save_json, 'w') as f:
        json.dump(dict_save, f, indent= 1)

def update_json(dict_agent, dict_level):
    for agent in dict_agent:
        for env_name in dict_level:
            if env_name not in dict_agent[agent]:
                dict_agent[agent][env_name] = {}
            for level in range(dict_level[env_name]['level_max']+1):
                if str(level) not in dict_agent[agent][env_name]:
                    dict_agent[agent][env_name][level] = []
    save_json(f'{SHOT_PATH}agent_all.json', dict_agent)

if os.path.exists(f'{SHOT_PATH}level_game.json') == False:
    save_json(f'{SHOT_PATH}level_game.json', {})

if os.path.exists(f'{SHOT_PATH}agent_all.json') == False:
    save_json(f'{SHOT_PATH}agent_all.json', {})

dict_level = json.load(open(f'{SHOT_PATH}level_game.json'))
dict_agent = json.load(open(f'{SHOT_PATH}agent_all.json'))

def fix_player():
    for id in df_agent.index:
        state_agent = df_agent.loc[id, 'CHECK']
        if pd.isna(state_agent):
            player = df_agent['AGENT'][id]
            df_agent.loc[id, 'CHECK'] = 'CHECKING'
            df_agent.to_excel('A:\StateAgent.xlsx', index=False)
            agent_test = importlib.util.spec_from_file_location('Agent_player', f"A:\AutoTrain\GAME\Agent\{player}\Agent_player.py").loader.load_module()
            bool_check_agent, msg = check_agent(agent_test)
            if bool_check_agent == True: #Sửa lại file excel trạng thái
                df_agent.loc[id, 'CHECK'] = 'DONE'
                id_add_agent = len(df_run)
                df_run.loc[id_add_agent] = np.nan
                df_run['ID'][id_add_agent] = player
                df_run.to_excel('A:\State.xlsx', index=False)
                dict_agent[player] = {}
            else:
                df_agent.loc[id, 'CHECK'] = 'BUG'
                df_agent.loc[id, 'NOTE'] = str(msg)
                
            df_agent.to_excel('A:\StateAgent.xlsx', index=False)
            update_json(dict_agent, dict_level)
            save_json(f'{SHOT_PATH}agent_all.json', dict_agent)


def fix_env():
    for id in df_env.index:
        state_env = df_env.loc[id, 'CHECK']
        if pd.isna(state_env):
            env_name = df_env['ENV'][id]
            df_env.loc[id, 'CHECK'] = 'CHECKING'
            df_env.to_excel('A:\StateEnv.xlsx', index = False)
            env = importlib.util.spec_from_file_location('env', f"A:/GAME/base/{env_name}/env.py").loader.load_module()
            bool_check_env, msg = check_env(env)
            if bool_check_env == True:
                df_env.loc[id, 'CHECK'] = 'DONE'
                df_run[env_name] = np.nan
                df_run.to_excel('A:\State.xlsx', index=False)
            else:
                df_env.loc[id, 'CHECK'] = 'BUG'
                df_env.loc[id, 'NOTE'] = str(msg)

            df_env.to_excel('A:\StateEnv.xlsx', index=False)

            dict_level[env_name] = {"Can_Split_Level": 'False',
                                    "level_max": 0,
                                    "id_remove":[]}
            update_json(dict_agent, dict_level)
            save_json(f'{SHOT_PATH}level_game.json', dict_level)


update_json(dict_agent, dict_level)
fix_player()
fix_env()