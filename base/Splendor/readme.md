# State
    * [0:100]: Trạng thái của 100 thẻ:
        * 5 là thẻ đang mở trên bàn chơi
        * -1 là đang giữ trên tay (thẻ người chơi úp)
        * 0 là không nhìn thấy (ở trên tay người chơi khác hoặc trong chồng thẻ úp trên bàn)
        * 1, 2, 3, 4 là thẻ bản thân đã mua, lần lượt người chơi phía sau đã mua
    
    * [100:106]: Nguyên liệu trên bàn chơi theo thứ tự: Red blue green black white gold
    * [106:118], [118:130], [130:142], [142:154]: Thông tin của 4 người chơi:
        * [+0:+6]: Nguyên liệu thường
        * [+6:+11]: Nguyên liệu vĩnh viễn
        * [+11]: Điểm
    * [154]: là turn nhưng đã bỏ nên luôn có giá trị bằng 0
    * [155:160]: Nguyên liệu đã lấy trong turn
    * [161:164]: Lần lượt báo hiệu thẻ ẩn trên bàn các cấp có còn hay không (1 là còn, 0 là hết)
    * [164]: Game đã kết thúc hay chưa

# Action
    * [0]: Bỏ lượt
    * [1]: Chọn phase lấy nguyên liệu
    * [2]: Chọn phase úp thẻ
    * [3]: Chọn phase mua thẻ
    * [4:9]: Lần lượt là lấy red, blue, green, black, white
    * [9:99]: Úp thẻ theo id
    * [99:102]: Lần lượt là úp thẻ ẩn cấp 1, cấp 2, cấp 3
    * [102:192]: Mua thẻ theo id
    * [192:198]: Lần lượt là trả nguyên liệu red, blue, green, black, white. gold