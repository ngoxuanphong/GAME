# #Thứ tự ưu tiên các action
# #Thẻ taget

# 1. -> Lấy thẻ
# 2. Nếu số nguyên liệu đang có <=7 và lấy được và có nguyên liệu để mua thẻ taget -> Lấy 3 nguyên liệu
# 3. Nếu thẻ taget ở trên bàn và úp được -> úp thẻ
# 4. Nếu trên bàn có thể lấy được thẻ có nguyên liệu mặc định cần cho thẻ taget -> Lấy thẻ đó với số lượng nguyên liệu bỏ ra ít nhất
# 5. Nếu có nguyên liệu cần để mua thẻ taget thì sẽ lấy nguyên liệu đó -> Lấy nguyên liệu
# 6. Nếu có thẻ miễn phí trên bàn -> Lấy thẻ miễn phí
# 7. Nếu có nguyên liệu cho thẻ taget và trên bàn nguyên liệu đó có >= 4 nguyên liệu  -> lấy 2 nguyên liệu đó
# 8. Nếu tổng số nguyên liệu <= 8, và có thể lấy 2 nguyên liệu bất kỳ trên bàn -> Lấy 2 nguyên liệu
# 9. Nếu mua được nguyên liệu nào trên bàn thì lấy nguyên liệu đó
# 10. Lấy thẻ bất kỳ trên bàn với số lượng nguyên liệu bỏ ra ít nhất