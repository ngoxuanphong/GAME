import random
import numpy as np

random.seed(10)
import os
import sys
from setup import game_name,time_run_game
sys.path.append(os.path.abspath(f"base/{game_name}"))
from env import *

player = 'KPI_130922'  #Tên folder của người chơi
path_data = f'Agent/{player}/Data'
if not os.path.exists(path_data):
    os.mkdir(path_data)
path_save_player = f'Agent/{player}/Data/{game_name}_{time_run_game}/'
if not os.path.exists(path_save_player):
    os.mkdir(path_save_player)

def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))
    return sig

def silu(x, theda = 1.0):
    return x * sigmoid(theda *x)

def random_player(p_state, temp_file, per_file):
    a = getValidActions(p_state)
    a = np.where(a == 1)[0]
    act_idx = np.random.randint(0, len(a))
    return a[act_idx], temp_file, per_file

def neural_network(play_state, file_temp):
    if 55 < len(play_state) < 70 or len(play_state) > 250 : # TLMN , TLMN_v2 , CENTURY
        matran1 = np.matmul(play_state,file_temp[0])
        matran1 = 1 / (1 + np.exp(-matran1))
        matran21 = np.matmul(matran1,file_temp[1])
        matran21 *= (matran21 > 0)
        matran2 = np.matmul(matran21, file_temp[2])
        return matran2    
    elif 120 <len(play_state)  < 170:# SPLENDOr SPlendor_view_only
        matrix1 = play_state@file_temp[0]
        matrixRL1 = 1 / (1 + np.exp(-matrix1))
        matrix2 = matrixRL1@file_temp[1]
        matrixRL2 = 1 / (1 + np.exp(-matrix2))
        all_action_val = matrixRL2@file_temp[2]
        return all_action_val  
    elif 170 < len(play_state) < 250  : #SHERIFF 
        matran1 = np.matmul(play_state, file_temp[0])
        matran1 = silu(matran1, theda = 1.0)
        matran2 = np.matmul(matran1, file_temp[1])
        return matran2
    else :#SUSHIGO-main, MACHIKOR0
        matran1 = np.matmul(play_state, file_temp[0])
        matran1 = silu(matran1, theda = 1.0)
        matran2 = np.matmul(matran1, file_temp[1])
        return matran2


def playerRandom1(play_state,file_temp,file_per):
    a = getValidActions(play_state)
    a = np.where(a == 1)[0]
    if len(file_temp) < 2 :
        if len(play_state)<55 or 70<len(play_state)<120:# SushiGo #MACHIKORO :
            file_temp = [(np.random.random((len(play_state),100)))*2 -0.67,np.random.random((100,getActionSize()))]
        elif 55 < len(play_state) < 70 :    # TLMN , TLMN_v2 
            file_temp = [(np.random.random((len(play_state),80)))*2 -0.6,np.random.random((80, 50)),np.random.random((50,getActionSize()))]
        elif 120 <len(play_state)  <170:  # Splendor , Splendor_view_only
              file_temp = [(np.random.random((len(play_state),100)))*2 -0.6 ,np.random.random((100, 50)),np.random.random((50,getActionSize()))]
        elif 170 < len(play_state) < 250: # SHERIFF
            file_temp = [(np.random.random((len(play_state),200)))*2 -0.6,np.random.random((200,getActionSize()))]
        elif len(play_state) > 250  :   # CENTURY
            file_temp = [(np.random.random((len(play_state),300)))*2 -0.6,np.random.random((300, 150)),np.random.random((150,getActionSize()))]

    matran2 = neural_network(play_state, file_temp)


    max = 0
    action_max = a[random.randrange(len(a))]
    for act in a:
        if matran2[act] > max:
            max = matran2[act]
            action_max = act

    if getReward(play_state) == 1:
        if type(file_per[0]) == int:
            file_per = [file_temp]
        else:
            file_per.append(file_temp)

    return action_max,file_temp,file_per


def playerScore1(play_state,file_temp,file_per):
    a = getValidActions(play_state)
    a = np.where(a == 1)[0]

    if len(file_per) < 2:
        file_ = np.load(f'{path_save_player}CK_Matran.npy',allow_pickle=True)
        file_per = [file_, np.zeros(len(file_))]
    
    if len(file_temp) < 2:
        Rand = random.randrange(len(file_per[0]))
        file_temp = [file_per[0][Rand],Rand]
        
    matran2 = neural_network(play_state, file_temp[0])

    max = 0
    action_max = a[random.randrange(len(a))]
    for act in a:
        if matran2[act] > max:
            max = matran2[act]
            action_max = act

    if getReward(play_state) == 1:
        file_per[1][file_temp[1]] += 1.2
    if getReward(play_state) == 0:
        file_per[1][file_temp[1]] -= 0.8
    return action_max,file_temp,file_per











# Hàm test và train 
def test(play_state,file_temp,file_per):
    player = 'KPI_130922' #Tên folder
    path_save_player = f'Agent/{player}/Data/{game_name}_{time_run_game}/' #Path để đọc file Test
    #Sau đó đọc file test như bình thường
    a = getValidActions(play_state)
    a = np.where(a == 1)[0]
    
    if len(file_temp) < 2:
        file_temp = np.load(f'{path_save_player}CK_Win.npy',allow_pickle=True)

    matran2 = neural_network(play_state, file_temp)

    max_ = 0
    action_max = a[random.randrange(len(a))]
    
    for act in a:
        if matran2[act] > max_:
            max_ = matran2[act]
            action_max = act
    return action_max,file_temp,file_per


def train(number_tran):
    list_player= [playerRandom1]*3 + [random_player]*(getAgentSize()-3)
    kq, file_ = normal_main(list_player, 1500, [0])
    np.save(f'{path_save_player}CK_Matran.npy',file_)

    list_player= [playerScore1]*3 + [random_player]*(getAgentSize()-3)
    kq, file_2 = normal_main(list_player,3500, [0])
    matran_Win= file_2[0][np.where(file_2[1] == max(file_2[1]))[0][0]]
    np.save(f'{path_save_player}CK_Win.npy',matran_Win)
        
    for buoc3 in range(2*number_tran-1):
        list_player = [playerRandom1]*2 + [test] + [random_player]*(getAgentSize()-3)
        kq, file_ = normal_main(list_player,1500, [0])
        file_.append(file_2[0][np.where(file_2[1] == max(file_2[1]))[0][0]])
        np.save(f'{path_save_player}CK_Matran.npy',file_)
        list_player= [playerScore1]*3 + [random_player]*(getAgentSize()-3)
        kq, file_2 = normal_main(list_player,3500, [0])
        matran_Win= file_2[0][np.where(file_2[1] == max(file_2[1]))[0][0]]
        np.save(f'{path_save_player}CK_Win.npy',matran_Win)
    return matran_Win


