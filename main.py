import os
import sys
import json
import time
import numpy as np
import random 
import importlib.util
import multiprocessing
import functools
import itertools
import multiprocessing.pool
import warnings 
from system.mainFunc import *
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

def train_1_player(player):
    p1 = load_module_player(player)
    p1.train(100000)

@timeout(time_run_game)
def train_1_player_with_timeout(players):
    if len(players) == 1:
        p1 = load_module_player(player[0])
        p1.train(100000)
    else:
        print_raise('Train_1_player')

def create_dict(dict_save_win__, count__, lst_player):
    for id in range(len(lst_player)):
        player_name = lst_player[id]
        if player_name not in dict_save_win__[game_name]:
            dict_save_win__[game_name][player_name] = int(count__[id])
        else:
            dict_save_win__[game_name][player_name] += int(count__[id])
    return dict_save_win__

def load_module_fight(player, mode_test):
    if mode_test == 'Test':
        spec = importlib.util.spec_from_file_location('Agent_player', f"Agent/{player}/Agent_player.py")
    else:
        spec = importlib.util.spec_from_file_location('Agent_player', f"system/Agent/{player}/Agent_player.py")
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
        list_player.append(load_module_fight(player, 'Test'))

    if len(list_player) < game.amount_player():
        player_random_need = game.amount_player() - len(list_player)
        for i in range(player_random_need):
            list_player_name.append('player_random')
            list_player.append(load_module_fight('player_random', 'Test'))

    lst_players = [i.test for i in list_player]
    count,_ = game.normal_main(lst_players, number_of_matches, [0])
    print(list_player_name, ' | ', count)
    return count, list_player_name

def fight_multi_player(game, players):
    dict_save_win = {'Catan':{},
                    'Century':{},
                    'MachiKoro':{}, 
                    'TLMN':{}, 
                    'TLMN_v2':{}, 
                    'Sheriff':{},
                    'Splendor':{}, 
                    'Splendor_v2':{},
                    'SushiGo':{}, }
    if len(players) != 0:
        if len(players) >= game.amount_player():
            list_combination = list(itertools.combinations(players, game.amount_player()))
            for lst_player_play in list_combination:
                count, lst_player = fight(game, lst_player_play)
                dict_save_win = create_dict(dict_save_win, count, lst_player)
        else:
            count, lst_player = fight(game, players)
            dict_save_win = create_dict(dict_save_win, count, lst_player)
        print(dict_save_win[game_name])
        with open(f'{path_save_json_test_player}/data_test_{game_name}.json', 'w') as f:
            json.dump(dict_save_win, f)
        return dict_save_win
    else:
        print_raise('Test')
    
@timeout(time_run_game)
def train():
    if len(players) != 0:
        pool = multiprocessing.Pool(processes=len(players))
        for player in players:
            pool.apply_async(train_1_player,args=(player,))
        pool.close()
        pool.join()
    else: 
        print_raise('Train')

def fight_test_1_player(game, players):
    list_player = [load_module_fight(players[0], 'Test')]
    for i in range(1, len(players)):
        player = players[i]
        list_player.append(load_module_fight(player, 'Test_1_player'))
    lst_players = [i.test for i in list_player]
    count,_ = game.normal_main(lst_players, 1 , [0])
    return count

def test_1_player(game, players):
    if len(players) == 1:
        win, lose = 0,0
        print('Agent:', players[0])
        list_all_players = os.listdir('system/Agent/')
        # print(list_all_players)

        progress_bar(0, number_of_matches)
        for match in range(number_of_matches):
            lst_player_fight = players + list(np.random.choice(list_all_players, size = (game.amount_player() -1), replace = False))
            count = fight_test_1_player(game, lst_player_fight)
            progress_bar(match+1, number_of_matches)
            if count[0] == 0: lose += 1   
            else: win += 1
        print(f'\nThắng: {win}, Thua: {lose}')
    else:
        print_raise('Test_1_player')


if __name__ == '__main__':
    game = setup_game(game_name)
    print('GAME:',  game_name)
    if type_run_code == 'Train':
        train()
    if type_run_code == 'Test':
        fight_multi_player(game, players)
    if type_run_code == 'Train_1_player':
        train_1_player_with_timeout(players)
    if type_run_code == 'Test_1_player':
        test_1_player(game, players)
