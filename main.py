import os
import sys
import importlib.util
import multiprocessing
import functools
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
    p = importlib.util.spec_from_file_location('Agent_player', f"Agent/{player}/Agent_player.py")
    p1 = p.loader.load_module()
    return p1
def train_1_player(player):
    p1 = load_module_player(player)
    p1.train(1)

def fight():
    list_player = []
    for i in range(len(players)):
        player = players[i]
        p1 = load_module_player(player)
        list_player.append(p1.test)
    if len(list_player) < game.amount_player():
        player_random_need = game.amount_player() - len(list_player)
        for i in range(player_random_need):
            p1 = load_module_player('player_random')
            print(p1)
            list_player.append(p1.test)
    count, file_per = game.normal_main(list_player, 1000, [0])
    return count, file_per

@timeout(time_run_game)
def train():
    pool = multiprocessing.Pool(processes=len(players))
    for player in players:
        pool.apply_async(train_1_player,args=(player,))
    pool.close()
    pool.join()

if __name__ == '__main__':
    game = setup_game(game_name)
    if type_run_code == 'Train':
        train()
    if type_run_code == 'Test':    
        count = fight()
        print(count)
