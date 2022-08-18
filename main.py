#Player_random, normal main gồm 3 tham số, list_player, số trận, và 
from Agent.player_random import player_random
from setup import setup_game, setup_player
game = setup_game()
p1 = setup_player()

#Train của chị Trang
p1.train(1)
#Test thử với random
list_player= [p1.test]*1 + [player_random]*3
kq, file_2 = game.normal_main(list_player,1000, [0])

# list_player = [player_random]*game.amount_player()
# count, file_per = game.normal_main(list_player, 10, [0])
# print(count)
