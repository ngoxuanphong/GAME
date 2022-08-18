# Run code

  - Thay đổi **game** và **agent** ở **setup.py**
  ```
    game_name = 'game_name'
    player = 'folder_player_name'
   ```
  - Chạy hàm **main.py** để train và test với random
  
  - Trong hệ thống đã có sẵn code chị Trang để mọi người tham khảo

# Agent
  - Mỗi người là một folder, đặt tên theo quy tắc ***tên_ngàynạpcode*** ví dụ: *Phong_08_12_2022*
  - Trong mỗi folder sẽ có:
      - file **Agent_player.py**: Lưu thuật toán của mọi người(chỉ có hàm)
      - folder **Data**: Nơi lưu data của từng game 
  - **Agent_Player.py** Tham khảo folder Trang có sẵn trong hệ thống
      - Có 2 hàm ***bắt buộc***:
        - **train(n)** n là tham số đầu vào(1 là 10000 ván, ...)
        - **test()** là hàm đọc file để test với random và người chơi khác
      - ***Chú ý:*** Thêm đoạn code này vào trong file thuật toán để import game và khởi tạo path lưu và đọc data
        ```
          import os
          import sys
          from setup import game_name
          from setup import player
          sys.path.append(os.path.abspath(f"base/{game_name}"))
          from env import *

          path_save_player = f'Agent/{player}/Data/{game_name}/'
          if not os.path.exists(path_save_player):
              os.mkdir(path_save_player)
        ```
# Game
