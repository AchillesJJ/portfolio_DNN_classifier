# encoding: utf-8
import subprocess as sub
import datetime
import pymysql
import time
import os
from utility import *

# database address
jydb_address = "117.122.223.35"
jydb_user_id = "zpy"
jydb_user_pwd = "Z1pe1y1@zpy"
jydb_db_name = "jydb02"
# on-line mode
conn = pymysql.connect(jydb_address, jydb_user_id, jydb_user_pwd, jydb_db_name)
# file path
ism_path = './ismdatabak_training'

today = datetime_to_string(datetime.date.today())

# decide the start and end dates of data
if is_market_day(today, conn):
    end_date = former_market_date(today, 7, conn)
else:
    end_date = former_market_date(today, 8, conn)
start_date = former_market_date(end_date, 5, conn)
# change to ism generation day
start_date = string_to_datetime(start_date)
start_date -= datetime.timedelta(days=1)
end_date = string_to_datetime(end_date)
end_date -= datetime.timedelta(days=1)

# clear obsolete data
del_file(ism_path)

# download data
date = start_date
while date <= end_date:
    # check whether data exists
    pass
    # download
    file_name = "".join(datetime_to_string(date).split('-'))
    sub.call("scp -r root@10.1.14.2:/data/ismcom/" + file_name + \
             "/*ismlist.[0-9]*.[0-9]*.7.[0-9]*.[0-9]* ./ismdatabak_training", shell=True)
    sub.call("scp -r root@10.1.14.2:/data/ismcom/" + file_name + \
             "/*ismfactors.[0-9]*.[0-9]*.7.[0-9]*.[0-9]* ./ismdatabak_training", shell=True)

    date += datetime.timedelta(days=1)

# delete extra space and other character
for file_name in os.listdir(ism_path):
    new_name = file_name.strip()
    os.rename(os.path.join(ism_path, file_name), os.path.join(ism_path, new_name))







