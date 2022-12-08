## :globe_with_meridians: Env_state
*   [0:52] state các lá bài, giá trị 0, 1, 2, 3 là  ở trên tay người chơi 0, 1, 2, 3. Còn -1 là đã được đánh trên bàn.
*   [52]: Player id đang đến lượt.
*   [53:56]: Tình trạng bỏ vòng của 3 người chơi còn lại. 1 là chưa bỏ vòng. 0 là đã bỏ vòng.
*   [57]: Id Người đã đánh bộ bài trên bàn.
*   [58:60]: Kiểu bộ bài và điểm bộ bài:
    * Kiểu bộ bài (58):
        * 0: Nothing
        * 1, 2, 3, 4: Đơn, đôi, tam, quý
        * 5, 6, ..., 13: Các dây 3, 4, ..., 11 lá bài
        * 14: 3 đôi thông
        * 15: 4 đôi thông
    * Điểm bộ bài: Bằng điểm lá bài có giá trị nhất trong bộ bài
*   [60]: Phase. 0 là chọn kiểu bài, 1 là chọn độ lớn của bộ
*   [61]: Temp lưu lại kiểu bộ bài đã chọn ở phase 0
*   Các lá bài: 0 là 3 bích, 1 là 3 tép, ..., và to nhất là 51 - hai cơ.

## :globe_with_meridians: p_state
*   [0:52] state các lá bài. 0 là trên tay, -1 là đã đánh, 1 là không nhìn được.
*   [52], [53], [54]: Tình trạng bỏ vòng của 3 ông kia, 1 là chưa bỏ, 0 là bỏ rồi.
*   [55], [56], [57]: Số lá bài còn lại của 3 ông kia
*   [58], [59]: Kiểu bộ bài, điểm bộ bài.
*   [60]: Phase mấy
*   [61]: Kiểu bộ bài đã chọn ở phase 0

## :globe_with_meridians: Action
*   [0:16]: Chọn kiểu bộ bài
*   [16:68]: Chọn điểm của bộ bài