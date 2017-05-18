from logging_tools import  get_logger
from paths import get_run_dir

run_d = get_run_dir()
logger = get_logger(run_d)
logger.info('hello temp')
from temp2 import f1

f1()
