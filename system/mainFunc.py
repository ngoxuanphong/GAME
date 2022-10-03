import time
def progress_bar(progress, total):
    bar_long = 100
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


dict_game_for_player = {
    'Splendor' : ['An_270922', 'Dat_270922', 'Hieu_270922', 'Khanh_270922', 'NhatAnh_270922', 'Phong_270922','An_200922','Phong_200922', 'Dat_200922', 'Khanh_200922', 'NhatAnh_200922', 'Phong_130922','Khanh_130922','Dat_130922'],
    'Splendor_v2' : ['An_270922', 'Dat_270922', 'Hieu_270922', 'Khanh_270922', 'Phong_270922', 'An_200922','Phong_200922', 'Dat_200922', 'Khanh_200922', 'Khanh_130922','Dat_130922','Hieu_130922'],
    'TLMN' : ['An_270922', 'Dat_270922', 'Khanh_270922','An_200922','Phong_200922', 'Dat_200922', 'Khanh_200922', 'Khanh_130922','Dat_130922'],
    'TLMN_v2' : ['An_270922', 'Dat_270922', 'Hieu_270922', 'Khanh_270922', 'NhatAnh_270922', 'Phong_270922', 'An_200922','Phong_200922', 'Dat_200922', 'Khanh_200922', 'Khanh_130922','Dat_130922'],
    'Century' : ['An_270922', 'Dat_270922', 'Hieu_270922', 'Khanh_270922','Phong_270922', 'An_200922','Phong_200922', 'Khanh_200922', 'Hieu_130922', 'Khanh_130922', 'Dat_130922'],
    'Sheriff' : ['Phong_270922', 'Hieu_270922', 'Khanh_270922', 'An_200922','Phong_200922', 'Dat_200922', 'Khanh_200922', 'NhatAnh_200922', 'Dat_130922', 'Khanh_130922'],
    'MachiKoro' : ['An_270922', 'Dat_270922', 'Hieu_270922', 'Khanh_270922', 'Phong_270922', 'An_200922','Phong_200922', 'Dat_200922', 'Khanh_200922', 'NhatAnh_200922','Dat_130922', 'NhatAnh_130922'],
    'SushiGo' : ['An_270922', 'Dat_270922', 'Hieu_270922', 'Khanh_270922', 'Phong_270922', 'An_200922','Phong_200922', 'Dat_200922', 'Khanh_200922', 'NhatAnh_200922', 'Hieu_130922', 'Phong_130922','Khanh_130922','Dat_130922', 'NhatAnh_130922','An_130922']
}
