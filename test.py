from base.SushiGo.env import *

import random
def player_random1(player_state,file_temp,file_per):
    a = get_list_action(player_state)
    b = random.randint(0,len(a)-1)
    if check_victory(player_state) == 1:
        print('1 win')
    return a[b],file_temp,file_per

def player_random2(player_state,file_temp,file_per):
    a = get_list_action(player_state)
    b = random.randint(0,len(a)-1)
    if check_victory(player_state) == 1:
        print('2 win')
    return a[b],file_temp,file_per

def player_random3(player_state,file_temp,file_per):
    a = get_list_action(player_state)
    b = random.randint(0,len(a)-1)
    if check_victory(player_state) == 1:
        print('3 win')
    return a[b],file_temp,file_per

def player_random4(player_state,file_temp,file_per):
    a = get_list_action(player_state)
    b = random.randint(0,len(a)-1)
    if check_victory(player_state) == 1:
        print('4 win')
    return a[b],file_temp,file_per

def player_random5(player_state,file_temp,file_per):
    a = get_list_action(player_state)
    b = random.randint(0,len(a)-1)
    if check_victory(player_state) == 1:
        print('5 win')
    return a[b],file_temp,file_per

list_player = [player_random1, player_random2, player_random3, player_random4, player_random5]
for i in range(1):
    print('--------')
    print(normal_main(list_player, 10, [0]))
# print(amount_player())