import logging
import time
import sys
import os

logger = logging.getLogger('ism_logger')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/ism_classifier_log.txt')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)
