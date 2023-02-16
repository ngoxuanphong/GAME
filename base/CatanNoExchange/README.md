## :globe_with_meridians: ENV_state
*   [0:19] **Tài nguyên trên các ô đất**
*   [19] **Vị trí Robber**
*   [20:39]   **Số trên các ô đất**
*   [39:48]   **Các cảng**
*   [48:53]   **Tài nguyên ngân hàng**
*   [53:58]   **Thẻ dev bank**
*   [58:100]  **Thông tin người chơi 0**

    - [0:5] **Tài nguyên**
    - [5:10] **Tài nguyên**
    - [10] **Điểm**
    - [11:26] **Tài nguyên**
    - [26:31] **Tài nguyên**
    - [31:35] **Tài nguyên**
    - [35] **Số thẻ Knight đã dùng**
    - [36] **Con đường dài nhất** 
    - [37:42] **Tỉ lệ trao đổi với bank**

*   [100:142] **Thông tin người chơi 1**
*   [142:184] **Thông tin người chơi 2**
*   [184:226] **Thông tin người chơi 3**

*   [226] **Danh hiệu quân đội mạnh nhất**
*   [227] **Danh hiệu con đường dài nhất**
*   [228] **Tổng xúc xắc**
*   [229] **Phase**
*   [230] **Turn**
*   [231] **Điểm đặt thứ nhất**
*   [232] **Số tài nguyên trả do bị chia**
*   [233] **Đang dùng thẻ dev gì**
*   [234] **Số lần sử dụng thẻ dev**

*   [235:239] **Loại thẻ dev được sử dụng trong turn hiện tại**
*   [239:244] **Lượng nguyên liệu còn lại khi bị chia đầu game**
*   [244] **Người chơi đang action(không hẳn là người chơi chính)**
*   [245:249] **Số nguyên liệu đã lấy trong turn đầu game**
*   [249:254] **Tài nguyên đưa ra trong trade offer**
*   [254:259] **Tài nguyên yêu cầu trong trade offer**
*   [254:274] **Tài nguyên trong kho dự trữ của người chơi**
*   [184:226] **Thông tin người chơi 3**
*   [280] **End Game**

## :bust_in_silhouette: P_state
*   [0:114] **Tài nguyên trên các ô đất** mỗi ô 6 index nguyên liệu, Cây, Gạch, Cừu, Lúa, Đá, Sa mạc (0, 1) (19*6)
*   [114:133] **Vị trí Robber** 19 vị trí đặt (0, 1)
*   [133:342]   **Số trên các ô đất** (0, 1) (2 -> 12)*19
*   [342:396]   **Các cảng** (Cây, Gạch, Cừu, Lúa, Đá, 3:1)
*   [396:401]   **Tài nguyên ngân hàng** Dạng 0, 1
*   [401]   **Thẻ dev bank** Dạng 0, 1
*   [402:540]  **Thông tin cá nhân**
    - [0:5]: Tài nguyên
    - [5:10]: Thẻ dev
    - [10]: Điểm
    - [11:83]: Đường: 72 đường, (0, 1)
    - [83:102]: Nhà: 19 nhà (0, 1)
    - [102:121]: Thành phố: (0, 1)
    - [121]: Số thẻ knight đã dùng (sl)
    - [122]: Con đường dài nhất (sl)
    - [123:138]: Tỉ lệ trao đổi với Bank(với mỗi nguyên liệu, lần lượt là tỉ lệ 2, 3, 4. có 5 nguyên liệu) (0, 1)

* **Thông tin người chơi khác**: [540:654], [654:768], [768:882]
    -  [0]: Tổng tài nguyên(sl)
    -  [1]: Tổng số thẻ dev(sl)
    -  [2]: Điểm (sl)
    -  [3:75]: Đường
    -  [75:94]: Nhà
    -  [94:113]: Thành phố
    -  [113]: Số thẻ knight đã dùng
    -  [114]: Con đường dài nhất

*   [882:886]: Danh hiệu quân đội mạnh nhất (0, 1) (4 người, người đang chơi là index 0)
*   [886:890]: Danh hiệu con đường dài nhất (0, 1) (4 người)
*   [890:902]: Tổng xx (0, 1) (2 -> 12)
*   [186]: Nguyên liệu còn lại có thể lấy ở đầu game (sl)
*   [187]: Điểm đặt thứ nhất
*   [188]: Số tài nguyên phải bỏ do bị chia
*   [189]: Đang dùng thẻ dev gì
*   [190]: Số lần dùng thẻ dev
*   [191:195]: Loại thẻ dev được sử dụng trong turn hiện tại
*   [195:200]: Số nguyên liệu còn lại ở trong kho
*   [205:218]: Các phase, gồm 13 phase 0 -> 12, phase 12 là phase chọn lấy nguyên liệu từ kho
*   [218]: Tài nguyên đưa ra trong trade offer để trade với bank
*   [218]: EndGame



## :video_game: Action


'''
## :bust_in_silhouette: P_state
*   [0:19] **Tài nguyên trên các ô đất**
*   [19] **Vị trí Robber**
*   [20:39]   **Số trên các ô đất**
*   [39:48]   **Các cảng**
*   [48:53]   **Tài nguyên ngân hàng** Dạng 0, 1
*   [53]   **Thẻ dev bank** Dạng 0, 1
*   [54:96]  **Thông tin cá nhân**
    - [0:5]: Tài nguyên
    - [5:10]: Thẻ dev
    - [10]: Điểm
    - [11:26]: Đường
    - [26:31]: Nhà
    - [31:35]: Thành phố
    - [35]: Số thẻ knight đã dùng
    - [36]: Con đường dài nhất
    - [37:42]: Tỉ lệ trao đổi với Bank
* **Thông tin người chơi khác**: [96:125], [125:154], [154:183]
    -  [0]: Tổng tài nguyên
    -  [1]: Tổng số thẻ dev
    -  [2]: Điểm
    -  [3:18]: Đường
    -  [18:23]: Nhà
    -  [23:27]: Thành phố
    -  [27]: Số thẻ knight đã dùng
    -  [28]: Con đường dài nhất

*   [183]: Danh hiệu quân đội mạnh nhất
*   [184]: Danh hiệu con đường dài nhất
*   [185]: Tổng xx
*   [186]: Nguyên liệu còn lại có thể lấy ở đầu game
*   [187]: Điểm đặt thứ nhất
*   [188]: Số tài nguyên phải bỏ do bị chia
*   [189]: Đang dùng thẻ dev gì
*   [190]: Số lần dùng thẻ dev
*   [191:195]: Loại thẻ dev được sử dụng trong turn hiện tại
*   [195:200]: Số nguyên liệu còn lại ở trong kho
*   [205:218]: Các phase, gồm 13 phase 0 -> 12, phase 12 là phase chọn lấy nguyên liệu từ kho
*   [218]: Tài nguyên đưa ra trong trade offer để trade với bank
*   [218]: EndGame

'''