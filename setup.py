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
def fight():
    from Agent.Trang import Agent_player as p1
    from Agent.player_random import Agent_player as p2
    from Agent.player_random import Agent_player as p3
    from Agent.player_random import Agent_player as p4
    return [p1.test, p2.test, p3.test, p4.test]