import time
def progress_bar(progress, total):
    bar_long = 50
    percent = int(bar_long * (progress/float(total)))
    bar = '█'*percent + '-'*(bar_long - percent)
    print(f"\r|{bar}| {percent: .2f}% | {progress}/{total}", end = "\r")

def print_raise(mode):
    if mode == 'Train' or mode == 'Test':
        print('Chắc hẳn bạn vừa để quên cái gì đó thì phải 😢😢😢 !')
        time.sleep(1)
        print('Chơi game để giải trí sau những giờ học căng thẳng')
        time.sleep(1)
        print('Nhưng...')
        time.sleep(1)
        print('Bạn không cho ai vào chơi cùng thì sao game bắt đầu được nhỉ 😐')
        time.sleep(1)
        print('Phải bạn không, cùng sửa nhé😘😘😘')
        time.sleep(1)
        print('VÀO SETUP THÊM NGƯỜI CHƠI VÀO')
    else:
        print('Để tôi kể bạn nghe...')
        time.sleep(1)
        print('Thôi lười quá!!!')
        time.sleep(1)
        print('VÀO SETUP SỬA SỐ LƯỢNG NGƯỜI CHƠI ĐÊ')
        time.sleep(1)
        print('Ở CHẾ ĐỘ NÀY CHỈ HỖ TRỢ 1 NGƯỜI THÔI')


