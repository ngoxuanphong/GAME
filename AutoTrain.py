import sys, os, time
import numpy as np
import pandas as pd
import importlib.util
import multiprocessing
import functools
import multiprocessing.pool
import warnings 
warnings.filterwarnings('ignore')
import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning,NumbaExperimentalFeatureWarning, NumbaWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)
warnings.simplefilter('ignore', category=NumbaWarning)

from setup import game_name, time_run_game, path



def setup_game(game_name):
    spec = importlib.util.spec_from_file_location('env', f"A:/GAME/base/{game_name}/env.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module 
    spec.loader.exec_module(module)
    return module


def load_module_player(player):
    return  importlib.util.spec_from_file_location('Agent_player', f"A:/GAME/Agent/{player}/Agent_player.py").loader.load_module()


def CreateFolder(players, game_name, time_run_game):
    player = players[0]  #Tên folder của người chơi
    path_data = f'A:/GAME/Agent/{player}/Data'
    if not os.path.exists(path_data):
        os.mkdir(path_data)
    path_save_player = f'A:/GAME/Agent/{player}/Data/{game_name}_{time_run_game}/'
    if not os.path.exists(path_save_player):
        os.mkdir(path_save_player)
    return path_save_player


def test_data(game, p1, path_save_player, level):
    PerDataStartTrain = list(np.load(f'{path_save_player}Train.npy',allow_pickle=True))
    win, per = game.numba_main_2(p1.Agent, 1000, PerDataStartTrain, level)
    return win 


def train(game, path_save_player, level, p1, time_loop):
    PerDataStartTrain = p1.DataAgent()
    for time in range(time_loop):
        win, PerDataStartTrain = game.numba_main_2(p1.Agent, 10000, PerDataStartTrain, level)
        np.save(f'{path_save_player}Train.npy',PerDataStartTrain)
    np.save(f'{path_save_player}Train.npy',PerDataStartTrain)

def check_code(data_train, id_in_file, game, players):
    p1 = load_module_player(players[0])
    try:
        win, PerDataStartTrain = game.numba_main_2(p1.Agent, 100, p1.DataAgent(), 0)
        
        start_10000 = time.time()
        win, PerDataStartTrain = game.numba_main_2(p1.Agent, 10000, p1.DataAgent(), 0)
        end_10000 = time.time()
        return True, int(time_run_game/(end_10000 - start_10000))
    except:
        print('Người chơi', players, 'Đang bị bug đó')
        change_excel_by_id(data_train, id_in_file, 'ERROR')
        return False

def train_and_test(game, players, all_level, time_loop):
    print(players)
    p1 = load_module_player(players[0])
    path_save_player = CreateFolder(players, game_name, time_run_game)
    train(game, path_save_player, 0, p1, time_loop)
    for level in range(len(all_level)):
        print(all_level)
        win = test_data(game, p1, path_save_player, level)
        all_level[level] = win//10
    return all_level

def change_excel_by_id(data_train, id_in_file, msg):
    print(data_train['ID'][id_in_file], msg)
    data_train['STATE'][id_in_file] = msg
    data_train.to_excel(path, index= False)

def calculate_time(func):
    def inner1(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print('| Time to run code', end - start)
    return inner1

@calculate_time
def main(game_name):
    print('---------')
    data_train = pd.read_excel(path)
    df_copy = data_train.loc[data_train['STATE'] == 'COPIED']
    game = setup_game(game_name)

    id_train = df_copy['ID'].iloc[0]
    id_in_file = data_train.loc[data_train['ID'] == id_train].index[0]

    all_level = [int(level) for level in data_train[game_name][id_in_file].strip('][').split(',')]
    __TrainManyPlayers__(game_name)
    change_excel_by_id(data_train, id_in_file, 'CHECKING')
    check_agent, time_loop = check_code(data_train, id_in_file, game, [id_train])
    print('Time_loop', time_loop)
    if check_agent == True:
        change_excel_by_id(data_train, id_in_file, 'RUNNING')
        all_level = train_and_test(game, [id_train], all_level, time_loop)
        
        data_train[game_name][id_in_file] = all_level
        change_excel_by_id(data_train, id_in_file, 'SUCCESS')



def __TrainManyPlayers__(game_name_):
    # game = setup_game(game_name_)
    if len(sys.argv) >= 2:
        sys.argv = [sys.argv[0]]
    sys.argv.append(game_name_)


if __name__ == '__main__':
    for time_ in range(3):
        main(game_name)
# __TrainManyPlayers__('TLMN', 'Agent')

