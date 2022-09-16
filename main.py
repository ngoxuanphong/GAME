import os
import sys
import json
import importlib.util
import multiprocessing
import functools
import itertools
import multiprocessing.pool
import warnings 
warnings.filterwarnings('ignore')

from setup import *
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
    sys.path.append(os.path.abspath(f"base/{game_name}"))
    import env
    return env

def load_module_player(player):
    return  importlib.util.spec_from_file_location('Agent_player', f"Agent/{player}/Agent_player.py").loader.load_module()
    # p1 = p
    # return p1

def train_1_player(player):
    p1 = load_module_player(player)
    p1.train(100000)

@timeout(time_run_game)
def train_1_player_with_timeout(player):
    p1 = load_module_player(player)
    p1.train(100000)

def create_dict(dict_save_win__, count__, lst_player):
    for id in range(len(lst_player)):
        player_name = lst_player[id]
        if player_name not in dict_save_win__[game_name]:
            dict_save_win__[game_name][player_name] = int(count__[id])
        else:
            dict_save_win__[game_name][player_name] += int(count__[id])
    return dict_save_win__

def load_module_fight(player):
    spec = importlib.util.spec_from_file_location('Agent_player', f"Agent/{player}/Agent_player.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module 
    spec.loader.exec_module(module)
    return module
    
def fight(game, players):
    list_player = []
    list_player_name = []

    for i in range(len(players)):
        player = players[i]
        list_player_name.append(player)
        list_player.append(load_module_fight(player))

    if len(list_player) < game.amount_player():
        player_random_need = game.amount_player() - len(list_player)
        for i in range(player_random_need):
            list_player_name.append('player_random')
            list_player.append(load_module_fight('player_random'))

    lst_players = [i.test for i in list_player]
    # print(list_player)
    count,_ = game.normal_main(lst_players, 1000, [0])
    print(list_player_name, ' | ', count)
    return count, list_player_name

def fight_multi_player(game, players, dict_save_win_):
    if len(players) >= game.amount_player():
        list_combination = list(itertools.combinations(players, game.amount_player()))
        for lst_player_play in list_combination:
            count, lst_player = fight(game, lst_player_play)
            dict_save_win_ = create_dict(dict_save_win_, count, lst_player)
    else:
        count, lst_player = fight(game, players)
        dict_save_win_ = create_dict(dict_save_win_, count, lst_player)
    return dict_save_win_
    
@timeout(time_run_game)
def train():
    pool = multiprocessing.Pool(processes=len(players))
    for player in players:
        pool.apply_async(train_1_player,args=(player,))
    pool.close()
    pool.join()

if __name__ == '__main__':
    game = setup_game(game_name)
    print('GAME:',  game_name)
    dict_save_win = {'Catan':{},
                    'Century':{},
                    'MachiKoro':{}, 
                    'TLMN':{}, 
                    'TLMN_v2':{}, 
                    'Sheriff':{},
                    'Splendor':{}, 
                    'Splendor_v2':{},
                    'SushiGo':{}, }
    if type_run_code == 'Train':
        train()
    if type_run_code == 'Test':
        dict_save_win = fight_multi_player(game, players, dict_save_win)
        print(dict_save_win[game_name])
        with open(f'{path_save_json_test_player}/data_test_{game_name}.json', 'w') as f:
            json.dump(dict_save_win, f)
    if type_run_code == 'Train_1_player':
        train_1_player_with_timeout(players[0])