import time
def progress_bar(progress, total):
    bar_long = 100
    percent = int(bar_long * (progress/float(total)))
    bar = '█'*percent + '-'*(bar_long - percent)
    print(f"\r|{bar}| {percent: .2f}% | {progress}/{total}", end = "\r")

def print_raise(mode):
    if mode == 'Train' or mode == 'Test':
        print('Chắc hẳn bạn vừa để quên cái gì đó thì phải 😢😢😢 !')
        time.sleep(1)
        print('Chơi game để giải trí sau những giờ học căng thẳng')
        time.sleep(1)
        print('Nhưng...')
        time.sleep(1)
        print('Bạn không cho ai vào chơi cùng thì sao game bắt đầu được nhỉ 😐')
        time.sleep(1)
        print('Phải bạn không, cùng sửa nhé😘😘😘')
        time.sleep(1)
        print('VÀO SETUP THÊM NGƯỜI CHƠI VÀO')
    else:
        print('Để tôi kể bạn nghe...')
        time.sleep(1)
        print('Thôi lười quá!!!')
        time.sleep(1)
        print('VÀO SETUP SỬA SỐ LƯỢNG NGƯỜI CHƠI ĐÊ')
        time.sleep(1)
        print('Ở CHẾ ĐỘ NÀY CHỈ HỖ TRỢ 1 NGƯỜI THÔI')


dict_game_for_player = {
    'Splendor' : ['An_270922', 'Dat_270922', 'Hieu_270922', 'Khanh_270922', 'NhatAnh_270922', 'Phong_270922','An_200922','Phong_200922', 'Dat_200922', 'Khanh_200922', 'NhatAnh_200922', 'Phong_130922','Khanh_130922','Dat_130922'],
    'Splendor_v2' : ['An_270922', 'Dat_270922', 'Hieu_270922', 'Khanh_270922', 'Phong_270922', 'An_200922','Phong_200922', 'Dat_200922', 'Khanh_200922', 'Khanh_130922','Dat_130922','Hieu_130922'],
    'TLMN' : ['An_270922', 'Dat_270922', 'Khanh_270922','An_200922','Phong_200922', 'Dat_200922', 'Khanh_200922', 'Khanh_130922','Dat_130922'],
    'TLMN_v2' : ['An_270922', 'Dat_270922', 'Hieu_270922', 'Khanh_270922', 'NhatAnh_270922', 'Phong_270922', 'An_200922','Phong_200922', 'Dat_200922', 'Khanh_200922', 'Khanh_130922','Dat_130922'],
    'Century' : ['An_270922', 'Dat_270922', 'Hieu_270922', 'Khanh_270922','Phong_270922', 'An_200922','Phong_200922', 'Khanh_200922', 'Hieu_130922', 'Khanh_130922', 'Dat_130922'],
    'Sheriff' : ['Phong_270922', 'Hieu_270922', 'Khanh_270922', 'An_200922','Phong_200922', 'Dat_200922', 'Khanh_200922', 'NhatAnh_200922', 'Dat_130922', 'Khanh_130922'],
    'MachiKoro' : ['An_270922', 'Dat_270922', 'Hieu_270922', 'Khanh_270922', 'Phong_270922', 'An_200922','Phong_200922', 'Dat_200922', 'Khanh_200922', 'NhatAnh_200922','Dat_130922', 'NhatAnh_130922'],
    'SushiGo' : ['An_270922', 'Dat_270922', 'Hieu_270922', 'Khanh_270922', 'Phong_270922', 'An_200922','Phong_200922', 'Dat_200922', 'Khanh_200922', 'NhatAnh_200922', 'Hieu_130922', 'Phong_130922','Khanh_130922','Dat_130922', 'NhatAnh_130922','An_130922'],
    'Catan_v2': ['An_130922', 'Dat_270922', 'Hieu_130922', 'Hieu_270922', 'Khanh_200922', 'NhatAnh_130922', 'NhatAnh_200922', 'NhatAnh_270922', 'Phong_130922', 'Phong_200922', 'Phong_270922']
}

from numba.typed import List
import os
import numpy as np
def load_data_per2(list_all_players, game_name_):
    lst_data = []
    for name in list_all_players:
        path_data = f'system/Agent/{name}/Data/{game_name_}_79200/'
        file_name = os.listdir(path_data)[0]
        data_in_file = np.load(f'{path_data}/{file_name}', allow_pickle=True)
        if 'Dat' in name:
            lst_data.append([data_in_file['w1'],data_in_file['w2']])
        elif 'Phong_270922' in name:
            if game_name_ == 'Splendor_v2': id_model = 0
            if game_name_ == 'Century': id_model = 1
            if game_name_ == 'MachiKoro': id_model = 0
            if game_name_ == 'Splendor': id_model = 2
            if game_name_ == 'TLMN': id_model = 0
            if game_name_ == 'TLMN_v2': id_model = 2
            if game_name_ == 'SushiGo': id_model = 1
            if game_name_ == 'Sheriff': id_model = 2
            if game_name_ == 'Catan_v2': id_model = 0
            data_in_file = data_in_file[id_model][0][-1]
            lst_data.append(data_in_file)
        elif 'Phong_200922' in name:
            if game_name_ == 'SushiGo': id_model = 0
            if game_name_ == 'TLMN': id_model = 0
            if game_name_ == 'Splendor': id_model = 0
            if game_name_ == 'Century': id_model = 0
            if game_name_ == 'MachiKoro': id_model = 0
            if game_name_ == 'Splendor_v2': id_model = 1
            if game_name_ == 'TLMN_v2': id_model = 1
            if game_name_ == 'Sheriff': id_model = 1
            if game_name_ == 'Catan_v2': id_model = 0
            data_in_file = data_in_file[id_model][0][-1]
            lst_data.append(data_in_file)
        elif 'Phong_130922' in name:
            if game_name_ == 'Splendor' or game_name_ == 'Catan_v2':
                data_in_file = data_in_file[0][0][-1]
                lst_data.append(data_in_file)
            if game_name_ == 'SushiGo':
                data_Phong_130922 = List()
                for i in range(3):
                    data_Phong_130922.append(data_in_file[i][0][-1])
                data_Phong_130922.append(np.array([[2.]]))
                lst_data.append(data_Phong_130922)
        elif 'NhatAnh_130922' in name:
            mylist0 = List()
            mylist1 = List()
            [mylist0.append(data_in_file[ii][0][0].astype(np.float64)) for ii in range(len(data_in_file))]
            for ii in range(len(data_in_file)):
                data_in_file[ii][1] = data_in_file[ii][1].reshape(1, len(data_in_file[ii][1]))
            [mylist1.append(data_in_file[ii][1]) for ii in range(len(data_in_file))]
            mylist = []
            mylist.append(mylist0)
            mylist.append(mylist1)
            lst_data.append(mylist)
        elif 'Hieu' in name or 'Khanh' in name:
            mylist = List()
            for i in data_in_file:
                mylist.append(i)
            lst_data.append(mylist)
        elif 'NhatAnh_270922' in name:
            mylist0 = List()
            mylist1 = List()
            [mylist0.append(data_in_file[0][ii].astype(np.float64)) for ii in range(len(data_in_file[0]))]
            for ii in range(len(data_in_file[1])):
                data_in_file[1][ii] = data_in_file[1][ii].flatten().astype(np.float64)
            [mylist1.append(data_in_file[1][ii]) for ii in range(len(data_in_file[1]))]
            mylist = []
            mylist.append(mylist0)
            mylist.append(mylist1)
            lst_data.append(mylist)
        elif 'An_130922' in name:
            mylist = List()
            for i in range(len(data_in_file)):
                if i%2 == 0:
                    mylist.append(data_in_file[i])
                else:
                    mylist.append(np.array([[data_in_file[i]]]).astype(np.float64))
            lst_data.append(mylist)
        elif 'An_200922' in name:
            if len(data_in_file) == 2:
                mylist1 = List()
                for i in range(len(data_in_file[0])):
                    if (i-2)%3 == 0:
                        mylist1.append(np.array([[data_in_file[0][i]]]).astype(np.float64))
                    elif (i-1)%3 == 0:
                        mylist1.append(np.array([data_in_file[0][i]]))
                    else:
                        mylist1.append(data_in_file[0][i].astype(np.float64))

                mylist2 = List()
                mylist2.append(np.array([[data_in_file[1]]]).astype(np.float64))

                mylist = []
                mylist.append(mylist1)
                mylist.append(mylist2)

                lst_data.append(mylist)
            else:
                mylist1 = List()
                for i in range(len(data_in_file)):
                    if i == 0:
                        mylist1.append(data_in_file[i])
                    elif i == 1:
                        mylist1.append(np.array([[data_in_file[1]]]).astype(np.float64))
                    else:
                        mylist1.append(np.array([data_in_file[2]]))
                mylist = List()
                mylist.append(mylist1)

                lst_data.append(mylist)
        elif 'An_270922' in name:
            if len(data_in_file) == 2:
                # print('Them data')
                mylist1 = List()
                for i in range(len(data_in_file[0])):
                    if (i-2)%3 == 0:
                        mylist1.append(np.array([[data_in_file[0][i]]]).astype(np.float64))
                    elif (i-1)%3 == 0:
                        mylist1.append(np.array([data_in_file[0][i]]))
                    else:
                        mylist1.append(data_in_file[0][i].astype(np.float64))

                mylist2 = List()
                mylist2.append(np.array([[data_in_file[1]]]).astype(np.float64))

                mylist = []
                mylist.append(mylist1)
                mylist.append(mylist2)

                lst_data.append(mylist)
            else:
                mylist1 = List()
                for i in range(len(data_in_file)):
                    if i == 0:
                        mylist1.append(np.array(data_in_file[i]))
                    elif i == 1:
                        mylist1.append(np.array([[data_in_file[1]]]).astype(np.float64))
                    else:
                        mylist1.append(np.array([data_in_file[2]]))

                mylist = List()
                mylist.append(mylist1)

                lst_data.append(mylist)
        else:
            lst_data.append(data_in_file)
    return lst_data



# def test_1_player_fight(game, game_name_, number_of_matches, players):
#         start = time.time()
#         win, lose = 0,0
#         if type(players[0]) == str: print('Agent:', players[0])

#         list_all_players = dict_game_for_player[game_name_]
#         list_data = load_data_per2(list_all_players, game_name_)
#         list_func_player = dict_func_all_player[game_name_]

#         id_players_all = np.arange(len(list_all_players))
#         per_2__ = 0

#         if type(players[0]) == str:
#             module_player = load_module_fight(players[0], 'Test')
#             players_main = [module_player.test2]
#         else:
#             players_main = [players[0]]

#         for match in range(number_of_matches):
#             np.random.shuffle(id_players_all)
#             lst_player_fight = players_main + [list_all_players[id_players_all[i]] for i in range(game.amount_player()-1)]
#             data_players = [per_2__] + [list_data[id_players_all[i]] for i in range(game.amount_player()-1)]
#             list_func = [list_func_player[id_players_all[i]] for i in range(game.amount_player()-1)]

#             count, per_2__ = fight_test_1_player(game, lst_player_fight, data_players, list_func)

#             if type(players[0]) == str: progress_bar(match+1, number_of_matches)
#             if count[0] == 0: lose += 1   
#             else: win += 1
        
#         if type(players[0]) == str: 
#             print(f'\nThắng: {win}, Thua: {lose}')
#             end = time.time()
#             print(f'Thời gian test:{end - start: .2f}s', )
#         return [win, lose], per_2__

# def test_1_player(game_name_, players, number_of_matches):
#     game = setup_game(game_name_)
#     if len(sys.argv) >= 2:
#         sys.argv = [sys.argv[0]]
#     sys.argv.append(game_name_)
#     # print(sys.argv, 'tên thật', game_name_)
#     importlib.reload(Agent_full)
#     if type(players) != list:
#         players = [players]
#     if len(players) == 1:
#         count_kq, per_2__ = test_1_player_fight(game, game_name_, number_of_matches, players)
#         return count_kq, per_2__
#     else:
#         print_raise('Test_1_player')