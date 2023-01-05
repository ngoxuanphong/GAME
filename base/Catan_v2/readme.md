# State
    * [0:19]: Tài nguyên trên các ô đất: 0, 1, 2, 3, 4, 5 lần lượt là Cây, Gạch, Cừu, Lúa, Đá, Sa mạc
    * [19]: Vị trí quân Robber
    * [20:39]: Số trên các ô đất
    * [39:48]: Các cảng: 0, 1, 2, 3, 4, 5 lần lượt là Cây, Gạch, Cừu, Lúa, Đá, Cảng 3:1
    * [48:53]: Tài nguyên Bank dạng 0: không, 1: có
    * [53]: Thẻ dev Bank dạng 0 hoặc 1
    * [54:96]: Thông tin cá nhân
        - [+0:+5]: Tài nguyên
        - [+5:+10]: Thẻ dev
        - [+10]: Điểm
        - [+11:+26]: Đường
        - [+26:+31]: Nhà
        - [+31:+35]: Thành phố
        - [+35]: Số thẻ knight đã dùng
        - [+36]: Con đường dài nhất
        - [+37:+42]: Tỉ lệ trao đổi với Bank
    
    * Thông tin người chơi khác: [96:125], [125:154], [154:183]
        - [+0]: Tổng tài nguyên
        - [+1]: Tổng số thẻ dev
        - [+2]: Điểm
        - [+3:+18]: Đường
        - [+18:+23]: Nhà
        - [+23:+27]: Thành phố
        - [+27]: Số thẻ knight đã dùng
        - [+28]: Con đường dài nhất
    
    * [183]: Danh hiệu quân đội mạnh nhất
    * [184]: Danh hiệu con đường dài nhất
    * [185]: Tổng xx
    * [186]: Pha
    * [187]: Điểm đặt thứ nhất
    * [188]: Số tài nguyên phải bỏ do bị chia
    * [189]: Đang dùng thẻ dev gì
    * [190]: Số lần dùng thẻ dev
    * [191:195]: Loại thẻ dev được sử dụng trong turn hiện tại
    * [195]: Số lần trạo trade offer
    * [196:201]: Tài nguyên đưa ra trong trade offer
    * [201:206]: Tài nguyên yêu cầu trong trade offer
    * [206:209]: Phản hồi của người chơi phụ
    * [209]: Người chơi chính
    * [210]: Game đã kết thúc hay chưa (1 là kết thúc rồi)


# Action
    * [0:54] Các action chọn điểm
    * [54]: Đổ xx
    * [55:59]: Lần lượt là dùng thẻ dev "Knight", "Roadbuilding", "Yearofplenty", "Monopoly"
    * [59:64]: Các action có thể mang lại sự "tăng" nguyên liệu
    * [64:83]: Các action chọn ô (khi di chuyển Robber)
    * [83:86]: Các action chọn người chơi để thực hiện tương tác
    * [86]: Mua đường
    * [87]: Mua nhà
    * [88]: Mua thành phố
    * [89]: Mua thẻ dev
    * [90]: Trade với người
    * [91]: Trade với bank
    * [92]: Kết thúc lượt
    * [93]: Người chơi phụ từ chối trade
    * [94]: Người chơi phụ đồng ý trade
    * [95:100]: Các action có thể mang lại sự "giảm" nguyên liệu
    * [100:103]: Người chơi chính duyệt trade của 3 người còn lại
    * [104]: Action dừng thêm nguyên liệu vào trade
    * [105]: Action bỏ qua trade (người chơi chính)
    