# - coding: utf-8 -*-
# python 3.4+

import logging
import time
import os

GPU = True
ADAM_RATE = 0.0002
# WeightDecay = None
WeightDecay = 0.00001
SAVE_MODEL = True

RAND_IN_NUM = 100
EPOCH = 2000
BATCH = 50

IMG_SIZE = 28
DATA_DIR = './data/'
LOG_DIR = './log/'
LOG_NAME = LOG_DIR + 'log_' + time.strftime('%Y-%m-%d_%H-%M-%S') + '.log'
LOG_LEVEL = logging.DEBUG

##################################################
# logging setting

try:
    os.mkdir(LOG_DIR)
except OSError:
    pass

logging.basicConfig(level=LOG_LEVEL,
                    format='[%(levelname)s]   \t %(asctime)s \t%(message)s\t',
                    datefmt='%Y/%m/%d (%A) - %H:%M:%S',
                    filename=LOG_NAME,
                    filemode='a'
                    )

console = logging.StreamHandler()
console.setLevel(LOG_LEVEL)
formatter = logging.Formatter('[%(levelname)s]  \t%(message)s\t')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

logging.info("saving log in: %s" % LOG_NAME)