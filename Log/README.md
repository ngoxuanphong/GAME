n = 15: là số agent mặc định lưu của mỗi level
LMAX   : Là level max
L      : level hiện tại
CDS    : Điều kiện pass level
1 là đã train, 0 là chưa train, -1 là đang train

Nếu một game, mà không có agent nào của level này, thì không có agent để copy thì làm sao?
# level_game.json:  lưu thông tin của các env theo level
    - Khi file level_game_all đã có level mới(L). Lần lượt train các agent lưu trữ ở L cho (L-1)
    lưu tên, trạng thái, tên vào index phía dưới(3, 4, 5). 
    - Khi đã train đủ N agent: lưu các agent có kết quả test cao nhất mà pass qua CDS(0, 1, 2)
    - Không đủ agent thì copy thêm(0, 1, 2). Để ý trạng thái của các agent copy tại đây
    - Khi trạng thái [0] đã sẵn sàng thì sửa "level_max" ở game này từ (L-1)thành L
    - Game name:
        - str(L)
            - [[], [], [], [], [], []]
            - 0: Trạng thái của agent đã lưu của các level này
            - 1: kết quả test của các agent
            - 2: tên các agent đã lưu của các level này
            - 3: trạng thái của N agent
            - 4: Tỉ lệ thắng N agent
            - 5: tên của N agent
        
# level_game_all.json: 
    - "level_max": LMAX đang có(level đang chờ agent đủ điều kiện)
    - str(LMAX)["Can Train"] == "True" thì LMAX += 1
    - str(L):
        - "Can Train": Có thể bắt đầu train cho các env được chưa
        - "Agents Name": [Tên các agent đang được lưu trữ ở level này]