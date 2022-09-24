from base.Splendor_v2.env import *
import time
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
import random
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


@njit()
def one_game(list_player, temp_file, per_file):
    env, lv1, lv2, lv3 = Reset()
    _cc = 0
    while env[100] <= 400 and _cc <= 10000:
        p_idx = env[100]%4
        p_state = get_player_state(env, lv1, lv2, lv3)
        act, temp_file[p_idx], per_file = list_player[p_idx](p_state, temp_file[p_idx], per_file)
        env, lv1, lv2, lv3 = step(act, env, lv1, lv2, lv3)

        if close_game(env) != 0:
            break

        _cc += 1
    
    turn = env[100]
    for i in range(4):
        env[100] = i
        act, temp_file[i], per_file = list_player[i](get_player_state(env, lv1, lv2, lv3), temp_file[i], per_file)
    
    env[100] = turn
    return close_game(env), per_file

@njit()
def player_random(p_state, temp_file, per_file):
    arr_action = get_list_action(p_state)
    act_idx = np.random.randint(0, len(arr_action))
    return arr_action[act_idx], temp_file, per_file
    
@njit()
def normal_main(list_player, num_game, temp_file, per_file):
    if len(list_player) != 4:
        print('Game chỉ cho phép có đúng 4 người chơi')
        return [-1,-1,-1,-1,-1], per_file
    
    num_won = [0,0,0,0,0]
    p_lst_idx = np.array([0,1,2,3])
    # temp_file = [np.array([0]),np.array([1]),np.array([2]),np.array([3])]
    for _n in range(num_game):
        np.random.shuffle(p_lst_idx)
        temp_file_new = [temp_file[p_lst_idx[i]] for i in range(amount_player())]
        list_player_new = [list_player[p_lst_idx[i]] for i in range(amount_player())]
        # print(temp_file_new, p_lst_idx)
        winner,per_file = one_game(
            list_player_new, temp_file_new, per_file,
        )
        # temp_file = [temp_file_new[np.where(p_lst_idx == i)[0][0]] for i in range(amount_player())]
        # print(temp_file)
        if winner != 0:
            num_won[p_lst_idx[winner-1]] += 1
        else:
            num_won[4] += 1

    return num_won, per_file


file_temp = [(np.random.random((30,100)))*2 -0.67,np.random.random((100,20))]
temp_file = file_temp*4
list_player = [player_random]*4
print(normal_main(list_player, 1, temp_file, [0]))

@njit()
def sigmoid(x:np.float64) -> np.float64:
    sig = 1 / (1 + np.exp(-x))
    return sig

@njit()
def silu(x:np.float64, theda = 1.0) -> np.float64:
    return x * sigmoid(theda *x)
    
@njit()
def neural_network(play_state:np.float64, file_temp) -> np.float64:
    # play_state = np.array(play_state)
    # if 55 < len(play_state) < 70 or len(play_state) > 300 : # TLMN , TLMN_v2 , CENTURY
        # matran1 = np.dot(play_state,file_temp[0])
        # matran1 = np.array(1 / (1 + np.exp(-matran1)))
        # matran21 = np.dot(matran1,file_temp[1])
        # matran21 *= (matran21 > 0)
        # matran2 = np.dot(matran21, file_temp[2])
        # return matran2.astype(np.float64)
    # elif 120 <len(play_state)  < 170:# SPLENDOr SPlendor_view_only
    #     matrix1 = np.dot(play_state, file_temp[0])
    #     matrixRL1 = 1 / (1 + np.exp(-matrix1))
    #     matrix2 = np.dot(matrixRL1, file_temp[1])
    #     matrixRL2 = 1 / (1 + np.exp(-matrix2))
    #     all_action_val = np.dot(matrixRL2, file_temp[2])
    #     return all_action_val.astype(np.float64)  
    # elif 170 < len(play_state) < 250  : #SHERIFF 
        matran1 = np.dot(play_state, file_temp[0])
        matran1 = silu(matran1, theda = 1.0)
        matran2 = np.dot(matran1, file_temp[1])
        return matran2.astype(np.float64)
    # else :#SUSHIGO-main, MACHIKOR0
    #     matran1 = np.dot(play_state, file_temp[0])
    #     matran1 = silu(matran1, theda = 1.0)
    #     matran2 = np.dot(matran1, file_temp[1])
    #     return matran2.astype(np.float64)

@njit()
def random_player(p_state, temp_file, per_file):
    arr_action = get_list_action(p_state)
    act_idx = np.random.randint(0, len(arr_action))
    return arr_action[act_idx], temp_file, per_file
    
@njit()
def playerRandom1(play_state:np.float64,file_temp:list,file_per:list):
    # print(type(file_temp))
    a = get_list_action(play_state)
    file_temp = [(np.random.random((len(play_state),100)))*2 -0.6 ,np.random.random((100, 50)),np.random.random((50,amount_action()))]
    # print(file_temp, file_per)
    # raise 1
    matran2 = neural_network(play_state, file_temp)
    # max = 0
    # action_max = a[random.randrange(len(a))]
    # for act in a:
    #     if matran2[act] > max:
    #         max = matran2[act]
    #         action_max = act

    # check_vic = check_victory(play_state)
    # if check_vic == 1:
    #     if type(file_per[0]) == int:
    #         file_per = [file_temp]
    #     else:
    #         file_per.append(file_temp)
    return int(a[0]),file_temp,file_per
    
play_state = np.array([7.0, 7.0, 7.0, 7.0, 7.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0, 4.0, 2.0, 2.0, 4.0, 5.0, 0.0, 0.0, 3.0, 0.0, 2.0, 1.0, 0.0, 5.0, 0.0, 0.0, 0.0, 2.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 5.0, 3.0, 3.0, 3.0, 3.0, 2.0, 3.0, 3.0, 0.0, 3.0, 5.0, 5.0, 1.0, 0.0, 3.0, 0.0, 0.0, 7.0, 4.0, 3.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 4.0, 0.0, 0.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 3.0, 3.0, 4.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 12.0])


file_temp = file_temp = [(np.random.random((len(play_state),100)))*2 -0.6 ,np.random.random((100, 50)),np.random.random((50,amount_action()))]
list_player= [playerRandom1]*4
temp_file = file_temp*4
kq, file_ = normal_main(list_player, 1, temp_file, np.array([0]))
print(kq)