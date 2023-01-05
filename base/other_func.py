# from numba import njit, jit
# @njit()
def progress_bar(progress, total):
    bar_long = 100
    percent = int(bar_long * (progress/float(total)))
    bar = '█'*percent + '-'*(bar_long - percent)
    print(f"\r|{bar}| {percent: .2f}% | {progress}/{total}\r", end = "\r")

