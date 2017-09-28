from os import path, makedirs
from datetime import datetime

weight_path = r'./trained_weights/'
patches = r'./patches/'
Labels_Path = r"seg/"
runs_dir = r'./runs/'
Src_Path = r"./train/"
Data_Path = r"data/"
WM_path = r"WM/"

run_dir = None

def get_run_dir():
    # create run folder
    global run_dir
    if run_dir is None:
        time = datetime.now().strftime('%d_%m_%Y_%H_%M')
        run_dir = runs_dir+time + '/'
        if not path.exists(run_dir):
            makedirs(run_dir)
    return run_dir