from Setup import get_list_action,amount_action,normal_main,check_victory,player_random
import random
import tensorflow as tf
import numpy as np
import time

PATH_DATA="Agent/Vang/Data/"

def NW(state,matrix):
    return np.dot(state,matrix)

def action_neraul_network(play_state,file_temp,file_per):
    a = get_list_action(play_state)
    if len(file_temp) == 1:
            file_temp = [0,0]
            column = len(play_state)
            row = random.randint(0,100)
            file_temp[0] = np.random.rand(column,row)*2-1
            file_temp[1] = np.random.rand(row,amount_action())*2-1
    list_score_1 = NW(play_state,file_temp[0])
    list_score_1 = tf.keras.activations.gelu(list_score_1)
    list_score_2 = NW(list_score_1,file_temp[1])
    max = -10000
    for action in a:
        if list_score_2[action] > max:
            max = list_score_2[action]
            b = action
    if check_victory(play_state) == 1:
        if file_per[0] == 0:
            file_per = []
        # print("file_per_ssss:",file_temp)
        file_per.append(file_temp)
    return b,file_temp,file_per

def action_NN_kick(play_state,file_temp,file_per):
    a = get_list_action(play_state)
    if file_per[0] == 0:
      file_per[0] = np.load(f'{PATH_DATA}Network_1.npy',allow_pickle=True)
    if len(file_temp) == 1:
        # file_temp = [0,0]
        template = random.randint(0,len(file_per[0])-1)
        file_temp = [template]
    else:
        template = file_temp[0]
    list_score_1 = NW(play_state,file_per[0][template][0])
    list_score_1 =tf.keras.activations.gelu(list_score_1)
    list_score_2 = NW(list_score_1,file_per[0][template][1])
    max = -10000
    for action in a:
        if list_score_2[action] > max:
            max = list_score_2[action]
            b = action
    result = check_victory(play_state)
    if len(file_per) == 1 and result != -1:
        file_per.append([1 for i in range(len(file_per[0]))])
        file_per.append([1 for i in range(len(file_per[0]))])
        
    if result == 1:
        file_per[1][template] +=3
    if result == 0:
        file_per[1][template] -=1
    if result != -1:
        file_per[2][template] +=1
    return b,file_temp,file_per

def action_NN_champion(play_state,file_temp,file_per):
    if len(file_temp) == 1:
        file_temp = np.load(f'{PATH_DATA}Vang_champion.npy',allow_pickle=True)
    a = get_list_action(play_state)
    list_score_1 = NW(play_state,file_temp[0])
    list_score_1 = tf.keras.activations.gelu(list_score_1)
    list_score_2 = NW(list_score_1,file_temp[1])
    max = -10000
    for action in a:
        if list_score_2[action] > max:
            max = list_score_2[action]
            b = action
    return b,file_temp,file_per

def func(x):
    a = [0,0,0,0,1]
    x *= 10000
    # while a[0] != max(a):
    print("1")
    a,b = normal_main([action_neraul_network,action_neraul_network,action_neraul_network,action_neraul_network,player_random],100,[0])
    # print(b)
    np.save(f'{PATH_DATA}Network_1.npy',b)
    print("2")
    a,b = normal_main([action_NN_kick,action_NN_kick,action_NN_kick,action_NN_kick,action_NN_kick],100,[0])
    t = np.array(b[1])/np.array(b[2])
    m = max(t)
    for i in range(len(t)):
        if b[1][i]/b[2][i] == m:
            print(i)
            np.save(f'{PATH_DATA}Vang_champion.npy',b[0][i])
    #   print("3")
    a,b = normal_main([action_NN_champion,action_NN_kick,action_NN_kick,player_random,player_random],100,[0])
    print(a)
func(1)
# a,b = normal_main([action_neraul_network,action_neraul_network,action_neraul_network,action_neraul_network,player_random],int(100),[0])
