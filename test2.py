from base.TLMN.env import *
import random as rd
import numpy as np

import os
import sys
from setup import game_name,time_run_game
sys.path.append(os.path.abspath(f"base/{game_name}"))
from env import *

player = 'Trang_130922'
path_data = f'Agent/{player}/Data'
if not os.path.exists(path_data):
    os.mkdir(path_data)
path_save_player = f'Agent/{player}/Data/{game_name}_{time_run_game}/'
if not os.path.exists(path_save_player):
    os.mkdir(path_save_player)


def player_random(p_state, temp_file, per_file):
    arr_action = getValidActions(p_state)
    act_idx = np.random.randint(0, len(arr_action))
    return arr_action[act_idx], temp_file, per_file

#  Buoc 1:
def player_Matran_Random(play_state,file_temp,file_per):
    a = getValidActions(play_state)
    # print(type(getValidActions(play_state)), 'Check_get_list_action')
    if len(file_temp) < 2:
        # print(play_state.dtype,'check_play_state')
        # print(getValidActions(play_state).dtype, 'Check_get_list_action')
        # print(type(getReward(play_state)),'cchecck_check_victoryv')
        # print(type(getActionSize()),'check_getActionSize()')
        file_temp = [(np.random.random((len(play_state),100)))*2 -0.6,np.random.random((100,getActionSize()))]
    Result_matran1 = np.matmul(play_state, file_temp[0])
    Result_matran1 *= Result_matran1 > 0
    Result_matran2 = np.matmul(Result_matran1, file_temp[1])
    max = 0
    action_max = a[rd.randrange(len(a))]
    for act in a:
        if Result_matran2[act] > max:
            max = Result_matran2[act]
            action_max = act
    # b = action_max
    
    if getReward(play_state) == 1:
        if type(file_per[0]) == int:
            file_per = [file_temp]
        else:
            file_per.append(file_temp)

    return action_max,file_temp,file_per

# Buoc 2:
def player_Matran_Score(play_state,file_temp,file_per):
    a = getValidActions(play_state)

    if len(file_per) < 2:
        file_ = np.load(f'{path_save_player}Trang_Matran.npy',allow_pickle=True)
        file_per = [file_, np.zeros(len(file_))]
    
    if len(file_temp) < 2:
        Rand = rd.randrange(len(file_per[0]))
        file_temp = [file_per[0][Rand],Rand]
    
    Result_matran1 = np.matmul(play_state, file_temp[0][0])
    Result_matran1 *= Result_matran1 > 0
    Result_matran2 = np.matmul(Result_matran1, file_temp[0][1])
    max = 0
    action_max = a[rd.randrange(len(a))]
    for act in a:
        if Result_matran2[act] > max:
            max = Result_matran2[act]
            action_max = act
    # b = action_max
    if getReward(play_state) == 1:
        file_per[1][file_temp[1]] += 1.2
    if getReward(play_state) == 0:
        file_per[1][file_temp[1]] -= 0.8
    return action_max,file_temp,file_per


# Buoc 3:
def test(play_state,file_temp,file_per):
    player = 'Trang_130922'
    path_save_player = f'Agent/{player}/Data/{game_name}_{time_run_game}/'
    a = getValidActions(play_state)
    
    if len(file_temp) < 2:
        file_temp = np.load(f'{path_save_player}Trang_Win.npy',allow_pickle=True)
        # file_temp = file[np.where(file_2[1] == max(file_2[1]))[0][0]]
    
    Result_matran1 = np.matmul(play_state, file_temp[0])
    Result_matran1 *= Result_matran1 > 0
    Result_matran2 = np.matmul(Result_matran1, file_temp[1])
    max_ = 0
    action_max = a[rd.randrange(len(a))]
    
    for act in a:
        if Result_matran2[act] > max_:
            max_ = Result_matran2[act]
            action_max = act
    # b = action_max

    return action_max,file_temp,file_per

#  Buoc 4: chay lai buoc 2
#  Ham tong hop 
def train(number_tran):
    # print(getAgentSize().dtype,'Check_getAgentSize()')
# number_tran la so nguyen duong >=1, neu muon chay X0.000 tran thi hay dien number_tran = x
    # Buoc1:
    list_player= [player_Matran_Random]*3 + [player_random]*(getAgentSize()-3)
    kq, file_ = normal_main(list_player, 1500, [0])
    np.save(f'{path_save_player}Trang_Matran.npy',file_)
    # Buoc2:
    list_player= [player_Matran_Score]*3 + [player_random]*(getAgentSize()-3)
    kq, file_2 = normal_main(list_player,3500, [0])
    # for i in kq:
    #     print(type(i), type(file_2), 'check_normal_main')
    matran_Win= file_2[0][np.where(file_2[1] == max(file_2[1]))[0][0]]
    np.save(f'{path_save_player}Trang_Win.npy',matran_Win)
        
    for buoc3 in range(2*number_tran-1):
        list_player = [player_Matran_Random]*2 + [test] + [player_random]*(getAgentSize()-3)
        kq, file_ = normal_main(list_player,1500, [0])
        file_.append(file_2[0][np.where(file_2[1] == max(file_2[1]))[0][0]])
        np.save(f'{path_save_player}Trang_Matran.npy',file_)
        list_player= [player_Matran_Score]*3 + [player_random]*(getAgentSize()-3)
        kq, file_2 = normal_main(list_player,3500, [0])
        matran_Win= file_2[0][np.where(file_2[1] == max(file_2[1]))[0][0]]
        np.save(f'{path_save_player}Trang_Win.npy',matran_Win)
    return matran_Win