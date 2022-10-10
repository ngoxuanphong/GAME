import numpy as np
from numba import njit
from numba.typed import List
import random
from base.SushiGo.env import get_list_action, amount_action, check_victory, amount_player, numba_main
@njit()
def random_player(p_state, temp_file, per_file):
    arr_action = get_list_action(p_state)
    arr_action = np.where(arr_action == 1)[0]
    act_idx = np.random.randint(0, len(arr_action))
    return arr_action[act_idx], temp_file, per_file

@njit()
def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))
    return sig

@njit()
def silu(x, theda = 1.0):
    return x * sigmoid(theda *x)

@njit()
def neural_network(play_state, file_temp):
    if 55 < len(play_state) < 70 or len(play_state) > 300 : # TLMN , TLMN_v2 , CENTURY
        matran1 = np.dot(play_state,file_temp[0])
        matran1 = 1 / (1 + np.exp(-matran1))
        matran21 = np.dot(matran1,file_temp[1])
        matran21 *= (matran21 > 0)
        matran2 = np.dot(matran21, file_temp[2])
        return matran2    
    elif 120 <len(play_state)  < 170:# SPLENDOr SPlendor_view_only
        matrix1 = np.dot(play_state, file_temp[0])
        matrixRL1 = 1 / (1 + np.exp(-matrix1))
        matrix2 = np.dot(matrixRL1, file_temp[1])
        matrixRL2 = 1 / (1 + np.exp(-matrix2))
        all_action_val = np.dot(matrixRL2, file_temp[2])
        return all_action_val  
    elif 170 < len(play_state) < 250  : #SHERIFF 
        matran1 = np.dot(play_state, file_temp[0])
        matran1 = silu(matran1, theda = 1.0)
        matran2 = np.dot(matran1, file_temp[1])
        return matran2
    else :#SUSHIGO-main, MACHIKOR0
        matran1 = np.dot(play_state, file_temp[0])
        matran1 = silu(matran1, theda = 1.0)
        matran2 = np.dot(matran1, file_temp[1])
        return matran2

@njit()
def playerRandom1(play_state,file_temp,file_per):
    a = get_list_action(play_state)
    a = np.where(a == 1)[0]
    if len(file_temp) < 2 :
        file_temp = List()
        if len(play_state)<55 or 70<len(play_state)<120:# SushiGo #MACHIKORO :
            file_temp.append((np.random.random((len(play_state),100)))*2 -0.67)
            file_temp.append(np.random.random((100,amount_action())))
        elif 55 < len(play_state) < 70 :    # TLMN , TLMN_v2 
            file_temp.append((np.random.random((len(play_state),80)))*2 -0.6)
            file_temp.append(np.random.random((80, 50)))
            file_temp.append(np.random.random((50,amount_action())))
        elif 120 <len(play_state)  <170:  # Splendor , Splendor_view_only
            file_temp.append((np.random.random((len(play_state),100)))*2 -0.6)
            file_temp.append(np.random.random((100, 50)))
            file_temp.append(np.random.random((50,amount_action())))
        elif 170 < len(play_state) < 250: # SHERIFF
            file_temp.append((np.random.random((len(play_state),200)))*2 -0.6)
            file_temp.append(np.random.random((200,amount_action())))
        else :   # CENTURY
            file_temp.append((np.random.random((len(play_state),300)))*2 -0.6)
            file_temp.append(np.random.random((300, 150)))
            file_temp.append(np.random.random((150,amount_action())))

    matran2 = neural_network(play_state, file_temp)

    max = 0
    action_max = a[random.randrange(len(a))]
    for act in a:
        if matran2[act] > max:
            max = matran2[act]
            action_max = act

    if check_victory(play_state) == 1:
        if len(file_per[0][0][0]) == 0:
            file_per = List()
            file_per.append(file_temp)
            # file_per = [file_temp]
        else:
            file_per.append(file_temp)

    return action_max,file_temp,file_per


@njit()
def playerScore1(play_state,file_temp,file_per):
    a = get_list_action(play_state)
    a = np.where(a == 1)[0]

    if len(file_temp) < 2:
        Rand = random.randrange(len(file_per[0]))
        file_temp = List()
        for matrix in file_per[0][Rand]:
            file_temp.append(matrix)
        file_temp.append(np.array([[Rand]]).astype(np.float64))

    matran2 = neural_network(play_state, file_temp)
    max = 0
    action_max = a[random.randrange(len(a))]
    for act in a:
        if matran2[act] > max:
            max = matran2[act]
            action_max = act
    check_vic = check_victory(play_state)
    if check_vic != -1:
        id_matrix = file_temp[-1][0][0]
        if check_vic == 1:
            file_per[1][0][0][0][int(id_matrix)] += 1.2
        else:
            file_per[1][0][0][0][int(id_matrix)] -= 0.8
    return action_max,file_temp,file_per

# Hàm test và train 2
@njit()
def test(play_state,file_temp,file_per):
    a = get_list_action(play_state)
    a = np.where(a == 1)[0]
    if len(file_temp) < 2:
        file_temp = file_per[0]

    matran2 = neural_network(play_state, file_temp)
    max_ = 0
    action_max = a[random.randrange(len(a))]
    
    for act in a:
        if matran2[act] > max_:
            max_ = matran2[act]
            action_max = act
    return action_max,file_temp,file_per


def train(number_tran):
    per_1_player = List()
    per_1_player.append(np.array([[]]))
    per = List()
    per.append(per_1_player)
    
    # list_player= [playerRandom1]*3 + [random_player]*(amount_player()-3)
    if amount_player() == 4:
        count, per = numba_main(playerRandom1, playerRandom1, playerRandom1, random_player,  1500, per)
    else:
        count, per = numba_main(playerRandom1, playerRandom1, playerRandom1, random_player, random_player,  1500, per)

    list21 = List()
    score_array = np.zeros((1,(len(per))))
    list21.append(score_array)
    list22= List()
    list22.append(list21)
    x = List()
    x.append(per)
    x.append(list22)

    # list_player= [playerScore1]*3 + [random_player]*(amount_player()-3)
    if amount_player() == 4:
        count, per2 = numba_main(playerScore1, playerScore1, playerScore1, random_player,  1500, x)
    else:
        count, per2 = numba_main(playerScore1, playerScore1, playerScore1, random_player, random_player, 1500, x)
    
    best_matrix = List()
    best_matrix.append(per2[0][np.argmax(per2[1][0][0][0])])

    # np.save(f'{path_save_player}CK_Win.npy',matran_Win)
    print('Xong lần đầu tiền')
    for buoc3 in range(2*number_tran-1):
        # list_player = [playerRandom1]*2 + [test] + [random_player]*(amount_player()-3)
        if amount_player() == 4:
            count, per = numba_main(playerRandom1, playerRandom1, test, random_player,  1500, best_matrix)
        else:
            count, per = numba_main(playerRandom1, playerRandom1, test, random_player, random_player, 1500, best_matrix)

        # list_player= [playerScore1]*3 + [random_player]*(amount_player()-3)
        list21 = List()
        score_array = np.zeros((1,(len(per))))
        list21.append(score_array)
        list22= List()
        list22.append(list21)
        x = List()
        x.append(per)
        x.append(list22)   

        if amount_player() == 4:
            count, per2 = numba_main(playerScore1, playerScore1, playerScore1, random_player,  1500, x)
        else:
            count, per2 = numba_main(playerScore1, playerScore1, playerScore1, random_player, random_player, 1500, x)

        best_matrix = List()
        best_matrix.append(per2[0][np.argmax(per2[1][0][0][0])])

        # np.save(f'{path_save_player}CK_Win.npy',matran_Win)
        print('Xong lần thứ', buoc3)
    return best_matrix