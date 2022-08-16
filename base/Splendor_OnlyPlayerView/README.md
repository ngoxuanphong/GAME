## Báo cáo SPLENDOR góc nhìn của mình
1.   `Tốc độ chạy`
      - **1000 Game**: 20s
      - **10000 Game**: 166s
2. `Chuẩn form`: Tạm thời test oke với random
3. `Đúng luật`: Tạm thời đúng
4. `Không bị loop vô hạn`: Đúng
5. `Các hàm player truyền vào`: 1000game: 56s
6. `Số ván check_vic > victory_thật`: 10000 ván thì có(thắng thật:2277, check_victory:2300)
7. `Giá trị state, action ở note`: 
8. `Tối thiểu số lần truyền vào player`: 250(từ 300-350)

## ENV_state
*   [0:90] các thẻ trên bàn: 5 là đang ở trên bàn, -(p_id) là đang úp, p_id là người chơi đã mua được
*   [100] Turn
*   [101:107] Nguyên liệu trên bàn, gồm có 6 nguyên liệu
*   [107 + 12 * p_id:119 + 12 * p_id] thông tin của người chơi, gồm có  6 nguyên liệu đang có, 5 nguyên liệu mặc định và điểm
*   [155:160] 5 Nguyên liệu mà người đó đã lấy trong turn
*   [161:164] 3 thẻ ẩn có thể úp cấp 1, 2, 3
*   [163] Phase

## P_state
*   [:6] là các nguyên liệu đang có trên bàn
*   [6: 18] thông tin của người chơi, gồm có  6 nguyên liệu đang có, 5 nguyên liệu mặc định và điểm
*   [18:102]:   12 thẻ bình thường trên bàn, mỗi thẻ có 7 state gồm: [điểm, loại thẻ, 5 nguyên liệu mua]
*   [102: 127]:   5 thẻ Noble trên bàn, mỗi thẻ có 5 state gồm: [5 loại nguyên liệu cần]
*   [127:148]:    3 thẻ úp trên tay, mỗi thẻ có 7 state gồm: [điểm, loại thẻ, 5 nguyên liệu mua]
*   [148: 153]:  5 nguyên liệu đã lấy trong phase lấy nguyên liệu
*   [153:156]: điểm của 3 người chơi còn lại

## action
* [0]   :Là action bỏ lượt, 
* [1:13] lấy 12 thẻ trên bàn
* [13:16] Là mở 3 thẻ đang úp
* [16:28] Úp 12 thẻ trên bàn
* [28:31] Úp 3 thẻ ẩn
* [31:36] Lấy 5 nguyên liêu
* [36:42] Trả 6 nguyên liệu
