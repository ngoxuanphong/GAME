# from numba import njit
# @njit()
def progress_bar(progress, total):
    bar_long = 100
    percent = int(bar_long * (progress/float(total)))
    bar = '█'*percent + '-'*(bar_long - percent)
    print(f"\r|{bar}| {percent: .2f}% | {progress}/{total}", end = "\r")


for i in range(1000):
    progress_bar(1, 1000)