import tensorflow as tf
import numpy as np
import pandas as pd
import preprocessing
from preprocessing import *
from utility import *
import random
import pymysql
import re
from sklearn.externals import joblib
from data_uploader import *
import subprocess as sub
import pipes
from data_downloader import *
from data_manager import *
import linecache

# ismlist = preprocessing.ism_list('./', 7)
# df = ismlist.ism_logit_and_labels('./ismlist.20180227')

# database address
jydb_address = "117.122.223.35"
jydb_user_id = "zpy"
jydb_user_pwd = "Z1pe1y1@zpy"
jydb_db_name = "jydb02"
# on-line mode
conn = pymysql.connect(jydb_address, jydb_user_id, jydb_user_pwd, jydb_db_name)
# 
# ls = ['125','1201','1491','1493','18721','34737']
# flag = "SELECT InnerCode, OpenPrice, ClosePrice FROM QT_DailyQuote WHERE " \
#        "InnerCode IN (" + ",".join(ls) + ") AND TradingDay='2018-03-08'"
# df = pd.read_sql(flag, conn)
# print(df)

# scaler_ls = joblib.load('standard_scaler_list.pkl')
# ism = ism_data('./ismdatabak_evaluation', batch_size=100, scaler_ls=scaler_ls, mode=2)
# logit, factor, ismcode = ism.evaluation_set()
# print(ismcode)

# ism = ism_list('./ismdatabak_training', 7)
# df = ism.ism_logit_and_labels('ismdatabak_training/ismlist.20180413.6.7.234.0001')
# print(df)

# ls = interval_filter('20180413', 55)
# for i in ls:
#     print(i)

# mg = manager('./data', mode=1)
# print(mg.get_date())

# line = linecache.getline('fuck.txt', 2)
# print(line)

# SF_map = SF_map()
# df = SF_map.SF_map()
# ls = list(set(list(df['FirstIndustryCode'].values)))
# print(len(ls))

# ism = ism_list('./new_ism', 7)
# df = ism.ism_logit_and_labels('new_ism/ismlist.20180404.6.7.234.0010')
# print(df)

print(former_market_date('2018-04-28', 2, conn))




