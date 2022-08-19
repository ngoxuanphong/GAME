#Player_random, normal main gồm 3 tham số, list_player, số trận, và 
import imp
from base.CENTURY.env import amount_player
from setup import setup_game, setup_player, fight
game = setup_game()
p1 = setup_player()
import warnings 
warnings.filterwarnings('ignore')

#Train của chị Trang
print('train')
print('So nguoi choi:', amount_player())
p1.train(1)

print('test')
list_player = fight()
count, file_per = game.normal_main(list_player, 1000, [0])
print(count)
