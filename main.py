#Player_random, normal main gồm 3 tham số, list_player, số trận, và 
from setup import setup_game, setup_player, fight
game = setup_game()
p1 = setup_player()

#Train của chị Trang
# p1.train(1)

list_player = fight()
count, file_per = game.normal_main(list_player, 10, [0])
print(count)
