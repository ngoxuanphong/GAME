import numpy as np
import itertools
import random
import copy
import time
import warnings 
from numba import njit
warnings.filterwarnings('ignore')

import os
import sys
from setup import game_name,time_run_game
sys.path.append(os.path.abspath(f"base/{game_name}"))
from env import *

player = 'Phong_200922'
path_data = f'system/Agent/{player}/Data'
if not os.path.exists(path_data):
    os.mkdir(path_data)
path_save_player = f'system/Agent/{player}/Data/{game_name}_{time_run_game}/'
if not os.path.exists(path_save_player):
    os.mkdir(path_save_player)


class Phong_att():
    def __init__(self):
        self.layer = 1
        self.count_matrix = 10
        self.count_test_matrix = 1
        self.add_reward = 1.2
        self.sub_reward = 0.8
        self.percent_matrix_choice = 0.8
        self.count_matrix_remove = 0.6


def player_random(state, temp, per):
    actions = get_list_action(state)
    action = random.choice(actions)
    return action, temp, per

def file_temp_to_action(state, file_temp, layer):
    a = get_list_action(state)
    RELU = np.ones(len(state))
    for i in range(layer -1):
        state1 = np.matmul(state,file_temp[0][i])
        state1 -= file_temp[1][i]
        RELU = state1*(state1>0)
    matrix_new = np.matmul(RELU,file_temp[0][-1])
    list_val_action = matrix_new[a]
    action = a[np.argmax(list_val_action)]
    return action
    

def activation_func(state,file_temp,file_per):
    if file_per == [0]:
        file_per = []
    if len(file_temp)  < 2:
        list_matrix = []
        list_bius = []
        for i in range(Phong_att().layer):
            list_matrix.append(np.random.random((len(state), len(state)))*2-Phong_att().count_matrix_remove)
            list_bius.append([random.uniform(-1, 1) for x in range(len(state))])
        list_matrix.append(np.random.random((len(state), amount_action()))*2-Phong_att().count_matrix_remove)
        file_temp = [list_matrix, list_bius]
    action = file_temp_to_action(state, file_temp, Phong_att().layer)
    if check_victory(state) == 1:
        if len(file_per) == 0:
            file_per = [file_temp]
        else:
            file_per.append(file_temp)
    return action,file_temp,file_per

def choose_matrix(state,file_temp,file_per, id_matrix):
    if len(file_temp) < 2:
        file_temp = [file_per[0][id_matrix], id_matrix]
    action = file_temp_to_action(state, file_temp[0], Phong_att().layer)

    check_vic = check_victory(state)
    if check_vic == 1:     
        file_per[1][file_temp[1]] += Phong_att().add_reward
    if check_vic == 0:     
        file_per[1][file_temp[1]] -= Phong_att().sub_reward
        
    return action,file_temp,file_per

def matrix1(state,file_temp,file_per):
    action,file_temp,file_per = choose_matrix(state,file_temp,file_per, file_per[2][0])
    return action,file_temp,file_per
def matrix2(state,file_temp,file_per):
    action,file_temp,file_per = choose_matrix(state,file_temp,file_per, file_per[2][1])
    return action,file_temp,file_per
def matrix3(state,file_temp,file_per):
    action,file_temp,file_per = choose_matrix(state,file_temp,file_per, file_per[2][2])
    return action,file_temp,file_per
def matrix4(state,file_temp,file_per):
    action,file_temp,file_per = choose_matrix(state,file_temp,file_per, file_per[2][3])
    return action,file_temp,file_per
def matrix5(state,file_temp,file_per):
    action,file_temp,file_per = choose_matrix(state,file_temp,file_per, file_per[2][4])
    return action,file_temp,file_per



def phong_create_list_matrix(file_per_save_data, type_train):
    while len(file_per_save_data) < amount_player():
        count, file_per_save_data = normal_main([activation_func]*amount_player(),Phong_att().count_matrix,[0])

    list_combinations = [list(i) for i in list(itertools.combinations(list(range(len(file_per_save_data))), amount_player()))]
    file_per_save_data = [file_per_save_data, [-99 for i in range(len(file_per_save_data))], []]
    for list_after_combinations in list_combinations:
        file_per_save_data[2] = list_after_combinations
        if amount_player() == 4:
            list_player = [matrix1, matrix2, matrix3, matrix4]
        elif amount_player() == 5:
            list_player = [matrix1, matrix2, matrix3, matrix4, matrix5]
        else:
            list_player = [matrix1, matrix2, matrix3, matrix4] + player_random*(amount_player()-1)
        count, file_per_save_data = normal_main(list_player, Phong_att().count_test_matrix,file_per_save_data)

    if type_train == 1:

        file_per_save_data[1] = np.array(file_per_save_data[1])
        file_per_save_data[0] = np.array(file_per_save_data[0])

        arg_sort = np.argsort(file_per_save_data[1])
        lst_argmax = arg_sort[int(len(file_per_save_data[1])*(Phong_att().percent_matrix_choice)):]
        return list(file_per_save_data[0][lst_argmax]), file_per_save_data[0][np.argmax(file_per_save_data[1])]

    if type_train == 2:
        return file_per_save_data[0][np.argmax(file_per_save_data[1])]

def save_file(check_point_save_file_all, p_check_point, p_bot_save):
    if check_point_save_file_all < p_check_point:
        check_point_save_file_all = p_check_point
        # print('checkpoint',check_point_save_file_all)
        np.save(f'{path_save_player}p_model_manh_nhat.npy', [p_bot_save])
    return check_point_save_file_all


def find_best_model(check_point_save_file_all):
    list_maxtrix_max_all = []

    p_check_point_1 = 1
    p_check_point_2 = 2
    p_check_point_3 = 3
    p_check_point_4 = 4
    p_check_point_5 = 5
    while len(list_maxtrix_max_all) < Phong_att().count_matrix:
        list_matrix_max_3 = []
        while len(list_matrix_max_3) < Phong_att().count_matrix:
            list_matrix_max_2 = []
            while len(list_matrix_max_2) < Phong_att().count_matrix:
                # print('            Tứ kết: ', len(list_matrix_max_2), end = '')
                list_matrix_max_1 = []
                # print(' Vòng bảng:', end = ' ')
                while len(list_matrix_max_1) < Phong_att().count_matrix:
                    # print(len(list_matrix_max_1), end = ' ')
                    file_per_save = []
                    list_file_per_save, bot_manh_nhat_temp = phong_create_list_matrix(file_per_save, 1)
                    check_point_save_file_all = save_file(check_point_save_file_all, p_check_point_1, bot_manh_nhat_temp) #checkpoint1, vòng bảng
                    list_matrix_max_1 += list_file_per_save

                list_file_per_save, bot_manh_nhat_temp = phong_create_list_matrix(list_matrix_max_1, 1)
                check_point_save_file_all = save_file(check_point_save_file_all, p_check_point_2, bot_manh_nhat_temp) #checkpoint2, tứ kết
                list_matrix_max_2 += list_file_per_save
                # print('')
            list_file_per_save, bot_manh_nhat_temp = phong_create_list_matrix(list_matrix_max_2, 1)
            list_matrix_max_3 += list_file_per_save
            check_point_save_file_all = save_file(check_point_save_file_all, p_check_point_3, bot_manh_nhat_temp) #checkpoint3, bán kết
            # print('      Bán kết: ', len(list_matrix_max_3))

        list_file_per_save, bot_manh_nhat_temp = phong_create_list_matrix(list_matrix_max_3, 1)
        check_point_save_file_all = save_file(check_point_save_file_all, p_check_point_4, bot_manh_nhat_temp) #checkpoint4, chung kết
        list_maxtrix_max_all += list_file_per_save
        # print('   Chung kết: ', len(list_maxtrix_max_all))
    # print('**************')
    bot_manh_nhat  = phong_create_list_matrix(list_maxtrix_max_all, 2)
    # print('Tìm xong nhà vô địch')
    check_point_save_file_all = save_file(check_point_save_file_all, p_check_point_5, bot_manh_nhat) #checkpoint5
    return bot_manh_nhat, check_point_save_file_all


def p_test_matrix_old(state,file_temp,file_per):
    file_temp = file_per[0]
    action = file_temp_to_action(state, file_temp, Phong_att().layer)
    return action,file_temp,file_per

def p_test_matrix_new(state,file_temp,file_per):
    file_temp = file_per[1]
    action = file_temp_to_action(state, file_temp, Phong_att().layer)
    return action,file_temp,file_per


# tối ưu ở đây, amount model = a
# sau khi tìm được ma trận mạnh hơn thì gán luôn
# thay vì chơi với random thì chơi với người a[0] + a[1] - a[2] - a[3]
# ở phần test thay vì thua thì đổi random thì đổi có chọn lọc
# hoặc có thể tính theo số trận thắng, nhiều hơn thì chọn model đó
# thêm numpy vào mốt số hàm có thể sử dụng được
def p_train_lan_2(n, a):
    if len(a) > 0:
        amount_model = len(a)
        p_len_state = len(a[0][0][0])
        model_manh_nhat = a
    else:
        model_manh_nhat = np.load(f'{path_save_player}p_model_manh_nhat.npy', allow_pickle=True)
        amount_model = len(model_manh_nhat)
        p_len_state = len(model_manh_nhat[0][0][0])
    model_manh_nhat_new = copy.deepcopy(model_manh_nhat)
    # print('len_model_manh_nhat', len(model_manh_nhat))
    np.save(f'{path_save_player}phong_id_model_choose.npy', np.zeros((len(model_manh_nhat), 2)))

    dict_model_vs_random = {}
    for phong_id_model_choose in range(len(model_manh_nhat)):
        dict_model_vs_random[phong_id_model_choose] = 0

    for change in range(n):
        for phong_id_model_choose in range(len(model_manh_nhat)):
            matrix_remove = (np.random.random((p_len_state, amount_action()))*2 - 1)*np.random.uniform(0.01, 0.09)
            model_manh_nhat_new[phong_id_model_choose][0][1] = model_manh_nhat[phong_id_model_choose][0][1] - matrix_remove
            list_player = [p_test_matrix_new, p_test_matrix_new, p_test_matrix_old, p_test_matrix_old] + [player_random]*int(amount_player()-4)
            count, _ = normal_main(list_player,1000,[model_manh_nhat[phong_id_model_choose], model_manh_nhat_new[phong_id_model_choose]])
            if (count[1] + count[0]) - ( count[3] + count[2]) >= 20:
                if count[1] > count[2] and count[1] > count[3] and count[0] > count[2] and count[0]>count[3]:
                    list_player = [p_test_matrix_new] + [player_random]*int(amount_player()-1)
                    count_, _= normal_main(list_player,1000,[model_manh_nhat[phong_id_model_choose], model_manh_nhat_new[phong_id_model_choose]])
                    # print(count,phong_id_model_choose,change,count_)
                    if count_[0] >= dict_model_vs_random[phong_id_model_choose] or (count[1] + count[0]) - ( count[3] + count[2]) > 60:
                        if count_[0] > 500:
                            # print('save', dict_model_vs_random)
                            dict_model_vs_random[phong_id_model_choose] = count_[0]
                            np.save(f'{path_save_player}p_model_manh_nhat.npy', model_manh_nhat_new)
                            model_manh_nhat = copy.deepcopy(model_manh_nhat_new)

def train(n):
    a = []
    check_point_save_file_all = 0
    start = time.time()
    for i in range(3):
        phong_find_best_model, check_point_save_file_all = find_best_model(check_point_save_file_all)
        end = time.time()
        a.append(phong_find_best_model)
        np.save(f'{path_save_player}p_model_manh_nhat.npy', a)
    # print(len(a), end - start)
    p_train_lan_2(n, a)
    # print('***Phong Train xong***')


def test(state,file_temp,file_per):
    if len(file_temp) < 2:
        player = 'Phong_200922'
        path_save_player = f'system/Agent/{player}/Data/{game_name}_{time_run_game}/'
        model_manh_nhat = np.load(f'{path_save_player}p_model_manh_nhat.npy', allow_pickle=True)
        phong_model_rate = np.load(f'{path_save_player}phong_id_model_choose.npy')
        id_choose = 99
        for id in range(len(phong_model_rate)):
            if phong_model_rate[id][0] + phong_model_rate[id][1] < 100:
                id_choose = id
                break
        if id_choose == 99:
            id_choose = np.argmax(phong_model_rate[:,0]/phong_model_rate[:,1])
            # print('id_choose',id_choose)
        file_temp = list(model_manh_nhat[id_choose])
        file_temp.append(phong_model_rate)
        file_temp.append(id_choose)
    action = file_temp_to_action(state, file_temp, Phong_att().layer)

    check_vic = check_victory(state)
    if check_vic != -1:
        if check_vic == 0:
            file_temp[2][file_temp[3]][1] += 1
        if check_vic == 1:
            file_temp[2][file_temp[3]][0] += 1
        player = 'Phong_200922'
        path_save_player = f'system/Agent/{player}/Data/{game_name}_{time_run_game}/'
        np.save(f'{path_save_player}phong_id_model_choose.npy', file_temp[2])
    return action,file_temp,file_per