import os
import sys
# lists_game = ['Splendor_OnlyPlayerView','CENTURY', 'MACHIKORO', 'SHERIFF', 'splendor', 'TLMN', 'TLMN_v2', 'SushiGo-main']
game_name = 'CENTURY'
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
    from Agent.player_random import Agent_player as p1
    from Agent.player_random import Agent_player as p2
    from Agent.player_random import Agent_player as p3
    from Agent.player_random import Agent_player as p4
    from Agent.player_random import Agent_player as p5
    return [p1.test, p2.test, p3.test, p4.test, p5.test]