import tensorflow as tf
import numpy as np
import pandas as pd
import preprocessing
from preprocessing import SF_map, ism_data
from utility import *
import random
import re

# ismlist = preprocessing.ism_list('./ismdatabak_training')
# # logit_labels = ismlist.ism_logit_and_labels('./ismdatabak_training/ismlist.20180313.1')
# logit_labels = ismlist.logit_and_labels()
# ismfactor = preprocessing.ism_factor('./ismdatabak_training')
# std_factor = ismfactor.standard_factor()
ismdata = preprocessing.ism_data('./ismdatabak_training', 10)
logit_batch, factor_batch, label_batch = ismdata.next_batch()
print(factor_batch)
print(label_batch)



