import os
import sys
game_name = 'Splendor_OnlyPlayerView'
player = 'Trang'
def setup_game():
    sys.path.append(os.path.abspath(f"base/{game_name}"))
    import env
    return env
def setup_player():
    sys.path.append(os.path.abspath(f"Agent/{player}"))
    import Agent_player as p1
    return p1
def fight_player():
    return 'Chưa làm xong'