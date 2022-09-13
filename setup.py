"""
 - players: list player muốn truyền vào
    - ở chế độ Train thì sẽ train đa luồng tương ứng số người chơi
    - chế độ Test nếu truyền vào không đủ người chơi thì sẽ tự động thêm random vào
 - games_name = ['Splendor_v2','Century', 'MachiKoro', 'Sheriff', 'Splendor', 'TLMN', 'TLMN_v2', 'SushiGo', 'Catan']
 - game_name: Tên game, có thể lấy ở games_name
 - time_run_game = Thời gian chạy game ở chế độ train, xong thời gian tự động dừng 
 - [Train, Test]
 - 'Train_1_player' để chạy mỗi người chơi, Không chạy đa luồng, dùng để test khi để code vào hệ thống 
"""



type_run_code = 'Train_1_player' #Train or Test or Train_1_player
players = ['Hieu_130922', 'Phong_130922']
game_name = 'SushiGo'
time_run_game = 15000
