import sys, os
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

from setup import game_name, time_run_game


def timeout(max_timeout):
    """Timeout decorator, parameter in seconds."""
    def timeout_decorator(item):
        """Wrap the original function."""
        @functools.wraps(item)
        def func_wrapper(*args, **kwargs):
            """Closure for function."""
            pool = multiprocessing.pool.ThreadPool(processes=1)
            async_result = pool.apply_async(item, args, kwargs)
            # raises a TimeoutError if execution exceeds max_timeout
            return async_result.get(max_timeout)
        return func_wrapper
    return timeout_decorator


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


def train_player(game, p1, PerDataStartTrain, path_save_player, level):
    for time in range(100):
        win, PerDataStartTrain = game.numba_main_2(p1.Agent, 10000, PerDataStartTrain, level)
        print('Lúc train', win, time)
        np.save(f'{path_save_player}Train.npy',PerDataStartTrain)
    np.save(f'{path_save_player}Train.npy',PerDataStartTrain)


def test_data(game, p1, path_save_player, level):
    PerDataStartTrain = list(np.load(f'{path_save_player}Train.npy',allow_pickle=True))
    win, per = game.numba_main_2(p1.Agent, 1000, PerDataStartTrain, level)
    print('test', win)
    return win 


@timeout(time_run_game)
def train(game, path_save_player, level, p1):
    PerDataStartTrain = p1.DataAgent()
    train_player(game, p1, PerDataStartTrain, path_save_player, level)


def train_and_test(game, players, all_level):
    print(players)
    p1 = load_module_player(players[0])
    path_save_player = CreateFolder(players, game_name, time_run_game)
    # try:
    train(game, path_save_player, 0, p1)
    # except:
    for level in range(len(all_level)):
        win = test_data(game, p1, path_save_player, level)
        all_level[level] = win//10
    return all_level

path = "C:\AutomaticTrain\State.xlsx"

def main(path):
    data_train = pd.read_excel(path)
    df_copy = data_train.loc[data_train['STATE'] == 'COPIED']
    game = setup_game(game_name)

    print(game_name)
    for i in df_copy.index:
            data_train = pd.read_excel(path)
            df_copy = data_train.loc[data_train['STATE'] == 'COPIED']
            id_train = df_copy['ID'].iloc[0]
            id_in_file = data_train.loc[data_train['ID'] == id_train].index[0]

            all_level = [int(level) for level in data_train[game_name][id_in_file].strip('][').split(',')]
            data_train['STATE'][id_in_file] = 'RUNNING'

            data_train.to_excel(path, index= False)
            all_level = train_and_test(game, [id_train], all_level)
            
            data_train[game_name][id_in_file] = all_level
            data_train['STATE'].iloc[id_in_file] = 'SUCCESS'
            data_train.to_excel(path, index= False)



def __TrainManyPlayers__(game_name_, player):
    # game = setup_game(game_name_)
    for game_name_ in ['TLMN_v2', 'TLMN', 'MachiKoro' ,'TLMN_v2', 'Splendor']:
        if len(sys.argv) >= 2:
            sys.argv = [sys.argv[0]]
            if 'Agent_player' in sys.modules:
                del sys.modules['Agent_player']
        sys.argv.append(game_name_)
        p1 = load_module_player(player)
        
        print(p1.getActionSize())

__TrainManyPlayers__('TLMN', 'Agent')

