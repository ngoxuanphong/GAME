# lists_game = ['Splendor_OnlyPlayerView','CENTURY', 'MACHIKORO', 'SHERIFF', 'splendor', 'TLMN', 'TLMN_v2', 'SushiGo-main']
import os
import sys
import importlib.util
import multiprocessing


def train_muil():
    pool = multiprocessing.Pool(processes=3)
    for player in players:
      pool.apply_async(train_1_player,args=(player))
    pool.close()
    pool.join()

games_name = ['TLMN']
players = ['Trang', 'Trang1', 'Trang2']


def setup_game(game_name):
    sys.path.append(os.path.abspath(f"base/{game_name}"))
    import env
    return env

def train_1_player(player):
    p = importlib.util.spec_from_file_location('Agent_player', f"Agent/{player}/Agent_player.py")
    p1 = p.loader.load_module()
    p1.train(1)

def train_player():
    global game_name
    global player
    for game_name in games_name:
        env = setup_game(game_name)
        # for player in players:
        #     print(player)
        #     train_1_player()
        if __name__ == '__main__':
            train_muil()

def fight():
    game = setup_game(game_name)
    from Agent.Trang import Agent_player as p1
    from Agent.Trang1 import Agent_player as p2
    from Agent.Trang2 import Agent_player as p3
    from Agent.player_random import Agent_player as p4
    from Agent.player_random import Agent_player as p5
    list_player = [p1.test, p2.test, p3.test, p4.test]
    count, file_per = game.normal_main(list_player, 1000, [0])
    return count, file_per