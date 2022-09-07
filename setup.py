"""
 - players: list player muốn truyền vào
    - ở chế độ Train thì sẽ train đa luồng tương ứng số người chơi
    - chế độ Test nếu truyền vào không đủ người chơi thì sẽ tự động thêm random vào
 - games_name = ['Splendor_OnlyPlayerView','CENTURY', 'MACHIKORO', 'SHERIFF', 'Splendor', 'TLMN', 'TLMN_v2', 'SushiGo-main']
 - game_name: Tên game, có thể lấy ở games_name
 - time_run_game = Thời gian chạy game ở chế độ train, xong thời gian tự động dừng 
 - [Train, Test] """

type_run_code = 'Test' #Train or Test
players = []
game_name = 'SushiGo-main'
time_run_game = 300
