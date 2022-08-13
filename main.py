from base.Splendor_OnlyPlayerView.env import *
from Agent.Trang.Agent import *
import random as rd

#Train của chị Trang
# Func(1)
# list_player= [player_Matran_Win,player_random,player_random,player_random]
# kq, file_2 = normal_main(list_player,1000, [0])

#Player_random, normal main gồm 3 tham số, list_player, số trận, và file_per
list_player = [player_random]*amount_player()
count, file_per = normal_main(list_player, 10, [0])
print(count)