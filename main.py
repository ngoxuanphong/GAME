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
from numba.typed import List
from system.mainFunc import print_raise, progress_bar, dict_game_for_player, load_data_per2
warnings.filterwarnings('ignore')
import warnings
import random
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning,NumbaExperimentalFeatureWarning, NumbaWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)
warnings.simplefilter('ignore', category=NumbaWarning)

from system import Agent_full
dict_func_all_player = {
    'Splendor': [Agent_full.test2_An_270922, Agent_full.test2_Dat_130922, Agent_full.test2_Hieu_270922, Agent_full.test2_Khanh_270922, Agent_full.test2_NhatAnh_270922, Agent_full.test2_Phong_130922, Agent_full.test2_An_200922, Agent_full.test2_Phong_130922, Agent_full.test2_Dat_130922, Agent_full.test2_Khanh_200922, Agent_full.test2_NhatAnh_200922, Agent_full.test2_Phong_130922, Agent_full.test2_Khanh_130922, Agent_full.test2_Dat_130922, ],
    'Splendor_v2': [Agent_full.test2_An_270922, Agent_full.test2_Dat_130922, Agent_full.test2_Hieu_270922, Agent_full.test2_Khanh_270922, Agent_full.test2_Phong_130922, Agent_full.test2_An_200922, Agent_full.test2_Phong_130922, Agent_full.test2_Dat_130922, Agent_full.test2_Khanh_200922, Agent_full.test2_Khanh_130922, Agent_full.test2_Dat_130922, Agent_full.test2_Hieu_130922, ],
    'TLMN': [Agent_full.test2_An_270922, Agent_full.test2_Dat_130922, Agent_full.test2_Khanh_270922, Agent_full.test2_An_200922, Agent_full.test2_Phong_130922, Agent_full.test2_Dat_130922, Agent_full.test2_Khanh_200922, Agent_full.test2_Khanh_130922, Agent_full.test2_Dat_130922, ],
    'TLMN_v2': [Agent_full.test2_An_270922, Agent_full.test2_Dat_130922, Agent_full.test2_Hieu_270922, Agent_full.test2_Khanh_270922, Agent_full.test2_NhatAnh_270922, Agent_full.test2_Phong_130922, Agent_full.test2_An_200922, Agent_full.test2_Phong_130922, Agent_full.test2_Dat_130922, Agent_full.test2_Khanh_200922, Agent_full.test2_Khanh_130922, Agent_full.test2_Dat_130922, ],
    'Century': [Agent_full.test2_An_270922, Agent_full.test2_Dat_130922, Agent_full.test2_Hieu_270922, Agent_full.test2_Khanh_270922, Agent_full.test2_Phong_130922, Agent_full.test2_An_200922, Agent_full.test2_Phong_130922, Agent_full.test2_Khanh_200922, Agent_full.test2_Hieu_130922, Agent_full.test2_Khanh_130922, Agent_full.test2_Dat_130922, ],
    'Sheriff': [Agent_full.test2_Phong_130922, Agent_full.test2_Hieu_270922, Agent_full.test2_Khanh_270922, Agent_full.test2_An_200922, Agent_full.test2_Phong_130922, Agent_full.test2_Dat_130922, Agent_full.test2_Khanh_200922, Agent_full.test2_NhatAnh_200922, Agent_full.test2_Dat_130922, Agent_full.test2_Khanh_130922, ],
    'MachiKoro': [Agent_full.test2_An_270922, Agent_full.test2_Dat_130922, Agent_full.test2_Hieu_270922, Agent_full.test2_Khanh_270922, Agent_full.test2_Phong_130922, Agent_full.test2_An_200922, Agent_full.test2_Phong_130922, Agent_full.test2_Dat_130922, Agent_full.test2_Khanh_200922, Agent_full.test2_NhatAnh_200922, Agent_full.test2_Dat_130922, Agent_full.test2_NhatAnh_130922, ],
    'SushiGo': [Agent_full.test2_An_270922, Agent_full.test2_Dat_130922, Agent_full.test2_Hieu_270922, Agent_full.test2_Khanh_270922, Agent_full.test2_Phong_130922, Agent_full.test2_An_200922, Agent_full.test2_Phong_130922, Agent_full.test2_Dat_130922, Agent_full.test2_Khanh_200922, Agent_full.test2_NhatAnh_200922, Agent_full.test2_Hieu_130922, Agent_full.test2_Phong_130922, Agent_full.test2_Khanh_130922, Agent_full.test2_Dat_130922, Agent_full.test2_NhatAnh_130922, Agent_full.test2_An_130922, ],
}

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
    spec = importlib.util.spec_from_file_location('env', f"base/{game_name}/env.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module 
    spec.loader.exec_module(module)
    return module

def load_module_player(player):
    return  importlib.util.spec_from_file_location('Agent_player', f"Agent/{player}/Agent_player.py").loader.load_module()

def train_1_player(player):
    p1 = load_module_player(player)
    a = time.time()
    print(a)
    p1.train(10)
    b = time.time()
    print(b)
    print(a-b)

@timeout(time_run_game)
def train_1_player_with_timeout(game, players):
    if len(players) == 1:
        p1 = load_module_player(players[0])
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
        if os.path.exists(f"Agent/{player}/Agent_player.py"):
            spec = importlib.util.spec_from_file_location('Agent_player', f"Agent/{player}/Agent_player.py")
        elif os.path.exists(f"system/Agent/{player}/Agent_player.py"):
            spec = importlib.util.spec_from_file_location('Agent_player', f"system/Agent/{player}/Agent_player.py")
        else:
            print(player)
            raise f'Không có player:'
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
        start = time.time()
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
        end = time.time()
        print(f'Thời gian test:{end - start: .2f}s', )

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

def fight_test_1_player(game, players, data_players, list_func):
    lst_players = [players[0]] + list_func
    count,per_2 = game.normal_main_2(lst_players, 1 ,data_players)
    return count, per_2[0]

def fight_test_1_player_2(game, players, data_players, list_func):
    if type(players[0]) == str:
        module_player = load_module_fight(players[0], 'Test')
        lst_players = [module_player.test]
    else:
        lst_players = [players[0]]
    for i in range(1, len(players)):
        player = players[i]
        module_player = load_module_fight(player, 'Test_1_player')
        lst_players.append(module_player.test)

    count,_ = game.normal_main(lst_players, 1 , [0])
    return count



def test_1_player_fight(game, game_name_, number_of_matches, players):
        start = time.time()
        win, lose = 0,0
        if type(players[0]) == str: print('Agent:', players[0])

        list_all_players = dict_game_for_player[game_name_]
        list_data = load_data_per2(list_all_players, game_name_)
        list_func_player = dict_func_all_player[game_name_]

        id_players_all = np.arange(len(list_all_players))
        per_2__ = 0

        if type(players[0]) == str:
            module_player = load_module_fight(players[0], 'Test')
            players_main = [module_player.test2]
        else:
            players_main = [players[0]]

        for match in range(number_of_matches):
            np.random.shuffle(id_players_all)
            lst_player_fight = players_main + [list_all_players[id_players_all[i]] for i in range(game.amount_player()-1)]
            data_players = [per_2__] + [list_data[id_players_all[i]] for i in range(game.amount_player()-1)]
            list_func = [list_func_player[id_players_all[i]] for i in range(game.amount_player()-1)]

            count, per_2__ = fight_test_1_player(game, lst_player_fight, data_players, list_func)

            if type(players[0]) == str: progress_bar(match+1, number_of_matches)
            if count[0] == 0: lose += 1   
            else: win += 1
        
        if type(players[0]) == str: 
            print(f'\nThắng: {win}, Thua: {lose}')
            end = time.time()
            print(f'Thời gian test:{end - start: .2f}s', )
        return [win, lose], per_2__

def test_1_player(game_name_, players, number_of_matches):
    game = setup_game(game_name_)
    if len(sys.argv) >= 2:
        sys.argv = [sys.argv[0]]
    sys.argv.append(game_name_)
    # print(sys.argv, 'tên thật', game_name_)
    importlib.reload(Agent_full)
    if type(players) != list:
        players = [players]
    if len(players) == 1:
        count_kq, per_2__ = test_1_player_fight(game, game_name_, number_of_matches, players)
        return count_kq, per_2__
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
        train_1_player_with_timeout(game, players)
    if type_run_code == 'Test_1_player':
        test_1_player(game_name, players, number_of_matches)

#splendor 447
#splendor_v2 250
#sushigo 103
#TLMN 106
#TLMN_v2 108, 2485