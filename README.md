# Clone and run code
  - ***Google colab***
    - Kết nối với **drive**
    - Tạo nơi muốn lưu hệ thống này(lưu data đã train lên drive để sau này sử dụng)
     ```
     %cd path_vừa_tạo
     %git clone https://github.com/ngoxuanphong/GAME.git
     %cd GAME
     ```
    - Chỉnh sửa trong setup.py
    ```
    !python main.py
    ```
    - chạy code bằng hàm 
    - **LƯU Ý**: 
      - Mỗi lần mở drive lên để chạy nhớ %cd đến thư mục và pull lại hệ thống vì có thể có update
      - Nhớ lưu ý đẩy folder của mình vào Agent, trong hệ thống đã có Agent mẫu
   
# Run code
   - players: list player muốn truyền vào
      - ở chế độ **Train** thì sẽ train đa luồng tương ứng số người chơi
      - chế độ **Test** nếu truyền vào không đủ người chơi thì sẽ tự động thêm random vào
   - games_name = ['Splendor_OnlyPlayerView','CENTURY', 'MACHIKORO', 'SHERIFF', 'splendor', 'TLMN', 'TLMN_v2', 'SushiGo-main']
   - game_name: Tên game, có thể lấy ở games_name
   - time_run_game = Thời gian chạy game ở chế độ train, xong thời gian tự động dừng 
   
  - Thay đổi **game**, **time_train** và **list_agent** ở **setup.py**
  ```
    type_run_code = 'Test' #'Train' or 'Test'
    players = ['Trang', 'Trang1'] 
    game_name = 'Splendor_OnlyPlayerView'
    time_run_game = 1000
   ```
  - Chạy hàm **main.py** để train và test với random
  
  - Trong hệ thống đã có sẵn code chị Trang để mọi người tham khảo

# Agent
  - Mỗi người là một folder, đặt tên theo quy tắc ***tên_ngàynạpcode*** ví dụ: *Phong_08_12_2022*
  - Trong mỗi folder sẽ có:
      - file **Agent_player.py**: Lưu thuật toán của mọi người(chỉ có hàm)
  - **Agent_Player.py** Tham khảo folder Trang có sẵn trong hệ thống
      - Có 2 hàm ***bắt buộc***:
        - **train(n)** n là tham số đầu vào(1 là 10000 ván, ...)
        - **test()** là hàm đọc file để test với random và người chơi khác
      - ***Chú ý:*** 
        - Trong thuật toán list_player **phải dựa vào amount_player()** tại vì các game thì số lượng người chơi khác nhau
        - player phải là tên **folder của người chơi**
        - **Thêm đoạn code này trong hàm Test**
          ```
            player = 'NhatAnh_0822' #Tên folder
            path_save_player = f'Agent/{player}/Data/{game_name}_{time_run_game}/' #Path để đọc file Test
            #Sau đó đọc file test như bình thường
          ```
        - Thêm đoạn code này vào trong file thuật toán để import game và khởi tạo path lưu và đọc data
        ```
        import os
        import sys
        from setup import game_name,time_run_game
        sys.path.append(os.path.abspath(f"base/{game_name}"))
        from env import *

        player = 'Trang'  #Tên folder của người chơi
        path_data = f'Agent/{player}/Data'
        if not os.path.exists(path_data):
            os.mkdir(path_data)
        path_save_player = f'Agent/{player}/Data/{game_name}_{time_run_game}/'
        if not os.path.exists(path_save_player):
            os.mkdir(path_save_player)
        ```
# Game
