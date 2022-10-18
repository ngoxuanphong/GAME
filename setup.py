"""
 - players: list player muốn truyền vào
    - ở chế độ Train thì sẽ train đa luồng tương ứng số người chơi
    - chế độ Test nếu truyền vào không đủ người chơi thì sẽ tự động thêm random vào
    - Test nếu truyền vào nhiều hơn số người chơi của game đó thì chơi tổ hợp và trả ra file json
 - games_name = ['Splendor_v2','Century', 'MachiKoro', 'Sheriff', 'Splendor', 'TLMN', 'TLMN_v2', 'SushiGo', 'Catan']
 - game_name: Tên game, có thể lấy ở games_name
 - time_run_game = Thời gian chạy game ở chế độ train, xong thời gian tự động dừng 
 - [Train, Test]
 - 'Train_1_player' để chạy mỗi người chơi, Không chạy đa luồng, dùng để test khi để code vào hệ thống 
 - 'Test_1_player' để test mỗi player với các agent bất kỳ mỗi game, return ra số trận thắng và thua
   'number_of_matches' Là số trận test khi dùng chế độ Test_1_player

 - Nâng cao dành cho người quản lý hệ thống:
      - path_save_json_test_player để lưu file json của khi test tổ hợp các player
"""



type_run_code = 'Train_1_player' #Train or Test or Train_1_player or Test_1_player
players = ['Phong_200922']
game_name = 'Catan_New'

time_run_game = 79200
 
number_of_matches = 1000 #Số trận test khi dùng khi test và test_1_player
path_save_json_test_player = '' #Nơi lưu file json data test các người chơi