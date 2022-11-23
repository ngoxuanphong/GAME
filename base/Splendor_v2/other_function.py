# def get_id_card(card_id):
#     if card_id < 40: return 'I', card_id +1
#     if 40 <= card_id < 70: return 'II', card_id - 39
#     if 70 <= card_id < 90: return 'III', card_id - 69
    
# def one_game_print(list_player, per_file, *print_mode):
#     env, lv1, lv2, lv3 = initEnv()
#     list_color = ['red', 'blue', 'green', 'black', 'white', 'auto_color']
#     def _print_():
#         print('----------------------------------------------------------------------------------------------------')
#         print('Lượt của người chơi:', env[100]%4 + 1, list_color)
#         print('B_stocks:', env[101:107], 'Turn:', env[100], )
#         print('Thẻ 1:', [i_+1 for i_ in get_list_id_card_on_lv(lv1)], list(lv1+1))
#         print('Thẻ 2:', [i_-39 for i_ in get_list_id_card_on_lv(lv2)], list(lv2-39))
#         print('Thẻ 3:', [i_-69 for i_ in get_list_id_card_on_lv(lv3)], list(lv3-69))
#         print('Noble:', [i_-89 for i_ in range(90,100) if env[:100][i_] == 5])
#         print('P1:', env[107:113], env[113:118], env[118], [get_id_card(i_) for i_ in range(90) if env[i_] == -1], [get_id_card(i_) for i_ in range(90) if env[i_] == 1],
#             '\nP2:', env[119:125], env[125:130], env[130], [get_id_card(i_) for i_ in range(90) if env[i_] == -2], [get_id_card(i_) for i_ in range(90) if env[i_] == 2],
#             '\nP3:', env[131:137], env[137:142], env[142], [get_id_card(i_) for i_ in range(90) if env[i_] == -3], [get_id_card(i_) for i_ in range(90) if env[i_] == 3],
#             '\nP4:', env[143:149], env[149:154], env[154], [get_id_card(i_) for i_ in range(90) if env[i_] == -4], [get_id_card(i_) for i_ in range(90) if env[i_] == 4],)
#         print('Nl đã lấy:', env[155:160],'Thẻ ẩn:', get_id_card(env[161]), get_id_card(env[162]), get_id_card(env[163]))
#         print('-------')

#     def _print_action_(act):
#         if act == 0:
#             print(f'Người chơi {p_idx+1} kết thúc lượt:', act)
#         elif act in range(1,13):
#             id_action = act-1
#             id_card_normal = get_id_card_normal_in_lv(lv1, lv2, lv3)
#             print(f'Người chơi {p_idx+1} mở thẻ trên bàn:', get_id_card(id_card_normal[id_action]),id_action,id_card_normal)
#         elif act in range(13,16):
#             id_action = act-13
#             id_card_normal = np.where(env[:90] == -(p_idx+1))[0]
#             print(f'Người chơi {p_idx+1} chọn mở thẻ đang úp:', get_id_card(id_card_normal[id_action]),id_action,id_card_normal)
#         elif act in range(16,28):
#             id_action = act-16
#             id_card_normal = get_id_card_normal_in_lv(lv1, lv2, lv3)
#             print(f'Người chơi {p_idx+1} chọn úp thẻ trên bàn:', get_id_card(id_card_normal[id_action]), id_action,id_card_normal)
#         elif act in range(28, 31):

#             print(f'Người chơi {p_idx+1} chọn úp thẻ ẩn:', get_id_card(env[161 + act-28]))
#         elif act in range(31, 36):
#             id_action = act-31
#             print(f'Người chơi {p_idx+1} lấy nguyên liệu:', list_color[id_action])
#         elif act in range(36, 42):
#             id_action = act-36
#             print(f'Người chơi {p_idx+1} trả nguyên liệu:', list_color[id_action])


#     temp_file = [[0],[0],[0],[0]]
#     _cc = 0
#     while env[100] <= 400 and _cc <= 10000:
#         p_idx = env[100]%4
#         p_state = getAgentState(env, lv1, lv2, lv3)
#         act, temp_file[p_idx], per_file = list_player[p_idx](p_state, temp_file[p_idx], per_file)
#         print('day la action he thong', act)
#         list_action = getValidActions(p_state)
#         if print_mode:
#             _print_()
#             for act_test in list_action:
#                 print(act_test, end = ' ')
#                 _print_action_(act_test)
#             print('________')
#             _print_action_(act)

#         if list_action[act] != 1:
#             raise Exception('Action không hợp lệ')

#         env, lv1, lv2, lv3 = stepEnv(act, env, lv1, lv2, lv3)
#         # print('Dây là lv1', lv1)
#         if checkEnded(env) != 0:
#             break

#         _cc += 1
    

#     turn = env[100]
#     for i in range(4):
#         env[100] = i
#         p_state = getAgentState(env, lv1, lv2, lv3)
#         p_state[161] = 1
#         act, temp_file[i], per_file = list_player[i](p_state, temp_file[i], per_file)
    
#     env[100] = turn
#     return checkEnded(env), per_file

# def normal_main_print(list_player, num_game=1, print_mode=False):
#     per_file = [0]
#     if len(list_player) != 4:
#         print('Game chỉ cho phép có đúng 4 người chơi')
#         return [-1,-1,-1,-1,-1], per_file
    
#     num_won = [0,0,0,0,0]
#     p_lst_idx = [0,1,2,3]
#     for _n in range(num_game):

#         # Shuffle người chơi
#         rd.shuffle(p_lst_idx)
#         if print_mode:
#             print('Thứ tự người chơi (thứ tự này sẽ ứng với P1,P2,P3,P4):', p_lst_idx)
#             print('Lưu ý: không phải người chơi index 0 là P1')

#         winner, per_file = one_game_print(
#             [list_player[p_lst_idx[0]], list_player[p_lst_idx[1]], list_player[p_lst_idx[2]], list_player[p_lst_idx[3]]], per_file, print_mode
#         )

#         if winner != 0:
#             num_won[p_lst_idx[winner-1]] += 1
#         else:
#             num_won[4] += 1

#     return num_won, per_file

