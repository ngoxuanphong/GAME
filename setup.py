"""
 - players: list player muốn truyền vào
    - ở chế độ train thì sẽ train đa luồng tương ứng số người chơi
    - chế độ test nếu truyền vào không đủ người chơi thì sẽ tự động thêm random vào
 - games_name = ['Splendor_OnlyPlayerView','CENTURY', 'MACHIKORO', 'SHERIFF', 'splendor', 'TLMN', 'TLMN_v2', 'SushiGo-main']
 - game_name: Tên game, có thể lấy ở games_name
 - time_run_game = Thời gian chạy game ở chế độ train, xong thời gian tự động dừng """

type_run_code = 'Train' #Train or test
players = ['Phong_08_12_2022', 'Trang1', 'Trang2']
game_name = 'TLMN_v2'
time_run_game = 1000
