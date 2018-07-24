# encoding: utf-8
#------------------------------------------------------------
#            ultility functions used by ISM class
#------------------------------------------------------------
import datetime
import numpy as np
import pandas as pd
import time
import sys
import os
import re

#----------------------------------------
#       date format transformation
#----------------------------------------
def datetime_to_string(dt):
    """
    transform datetime object to string "YY-MM-DD"
    """
    return dt.strftime("%Y-%m-%d")

def string_to_datetime(string):
    """
    transform string "YY-MM-DD" to datetime object
    """
    return datetime.date(*map(int, string.split('-')))

def timestamp_to_string(stamp):
    """
    transform time stamp to string "YY-MM-DD"
    """
    return str(stamp).split(' ')[0]

#----------------------------------------
#       date objects calculation
#----------------------------------------
def former_date(string, L_shift):
    """
    return the former date (L_shift days ago) in form "YY-MM-DD"
    """
    dt = string_to_datetime(string)
    dt -= datetime.timedelta(days=L_shift)
    return datetime_to_string(dt)
    
def future_date(string, R_shift):
    """
    return the future date (R_shift days later) in form "YY-MM-DD"
    """
    dt = string_to_datetime(string)
    dt += datetime.timedelta(days=R_shift)
    return datetime_to_string(dt)
    
def former_days(string, L_shift):
    """
    return all former days date as list with form "YY-MM-DD" 
    (from given date(string) to L_shift days ago)
    """
    dt_end = string_to_datetime(string)
    dt_start = dt_end - L_shift
    dtls = pd.date_range(dt_start, dt_end)
    return list(map(timestamp_to_string, dtls))
    
def future_days(string, R_shift):
    """
    return all future days date as list with form "YY-MM-DD" 
    (from given date(string) to R_shift days latter)
    """
    dt_start = string_to_datetime(string)
    dt_end = dt_start + datetime.timedelta(R_shift)
    dtls = pd.date_range(dt_start, dt_end)
    return list(map(timestamp_to_string, dtls))
    
def is_market_day(string, conn=None, df=None):
    """
    identify whether the given date(string) is a market day
    """
    # on-line mode
    date = string_to_datetime(string)
    if conn is not None:
        flag = "SELECT IfTradingDay FROM QT_TradingDayNew WHERE TradingDate=\'" + \
               string + "\' AND SecuMarket=83 LIMIT 1"
        is_market = int(pd.read_sql(flag, conn)['IfTradingDay'])
    else: # off-line mode
        is_market = int(df[df['TradingDate']==string]['IfTradingDay'])
        
    if is_market == 1:
        return True
    else:
        return False        
        
    
def former_market_date(string, L_shift, conn=None, df=None):
    """
    return the former market date(L_shift market days ago) in form "YY-MM-DD"
    """
    date = string_to_datetime(string)
    # on-line mode
    if conn is not None:
        cnt = 1
        while True:
            if cnt == L_shift or L_shift == 0:
                break
            date -= datetime.timedelta(days=1)
            if is_market_day(datetime_to_string(date), conn=conn):
                cnt += 1
    else: # off-line mode
        cnt = 1
        while True:
            if cnt == L_shift or L_shift == 0:
                break
            date -= datetime.timedelta(days=1)
            if is_market_day(datetime_to_string(date), df=df):
                cnt += 1
        
    return datetime_to_string(date)    
    
def future_market_date(string, R_shift, conn=None, df=None):
    """
    return the future market date(R_shift market days latter) in form "YY-MM-DD"
    but cannot exceed nowaday
    """
    date = string_to_datetime(string)
    # on-line mode
    if conn is not None:
        cnt = 1
        while True:
            if cnt == R_shift or R_shift == 0:
                break
            date += datetime.timedelta(days=1)
            if is_market_day(datetime_to_string(date), conn=conn):
                cnt += 1
    else: # off-line mode
        cnt = 1
        while True:
            if cnt == R_shift or R_shift == 0:
                break
            date += datetime.timedelta(days=1)
            if is_market_day(datetime_to_string(date), df=df):
                cnt += 1
    
    # check whether exceed date of today
    # nowaday = datetime.date.today()
    # if date < nowaday:
    #     return datetime_to_string(date)
    # else:
    #     print("Future market date has exceeded limit") # should use raise
    return datetime_to_string(date)
    

def market_days(string, L_shift, R_shift, conn):
    """
    return all market days in a list of string "YY-MM-DD" from start to end
    """
    # middle date
    dt_mid = string_to_datetime(string)
    # from former date to middle date
    dt = dt_mid
    market_days_ls = []
    cnt = 1
    while True:
        if cnt == L_shift or L_shift ==0:
            break
        dt -= datetime.timedelta(days=1)
        if is_market_day(datetime_to_string(dt), conn):
            cnt += 1
            market_days_ls.append(datetime_to_string(dt))
    market_days_ls.reverse() # reverse to normal order
    # from middle date to future date, no duplication of dt_mid
    market_days_ls.append(datetime_to_string(dt_mid))
    dt = dt_mid
    cnt = 1
    while True:
        if cnt == R_shift or R_shift == 0:
            break
        dt += datetime.timedelta(days=1)
        if is_market_day(datetime_to_string(dt), conn):
            cnt += 1
            market_days_ls.append(datetime_to_string(dt))
    
    return market_days_ls
    
#-----------------------------------------------------------
#                   List Operations
#-----------------------------------------------------------
def flatten(ls):
    """
    flatten nested list into a 1-D list
    """    
    return [x for y in ls for x in y]
    

#-----------------------------------------------------------
#                   File Operations
#-----------------------------------------------------------
def del_file(path):
    """
    delete all files in given directory with iteratin
    """
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)
            

def interval_filter(ism_path, interval, conn):
    """
    choose all data files with given interval from directory of ism_path
    """
    # delete extra space and other character
    for file_name in os.listdir(ism_path):
        new_name = file_name.strip()
        os.rename(os.path.join(ism_path, file_name), os.path.join(ism_path, new_name))
        
    file_name_ls = []
    for file_name in os.listdir(ism_path):
        f = re.match(r'ismlist\.(\d{4})-?(\d{2})-?(\d{2}).*', file_name)
        if f is None:
            continue
        else:
            # check the interval
            cnt = 0
            with open(os.path.join(ism_path, f.group()), 'r') as ism_file:
                for line in ism_file:
                    cnt += 1
                    if cnt == 2:
                        # parse the line
                        line = line.replace('\"', '')
                        line = line.replace('],[', '];[')
                        line = line.strip()
                        line = line.replace(']', '')
                        line = line.replace('[', '')
                        line = line.split(';')
                        line[0] = line[0].split('|')
                        ls = list(map(lambda x: x.split(','), line[1:]))
                        ls = list(map(lambda x: list(map(lambda y: y.split('|'), x)), ls))
                        line[1:] = ls
                        ProRunDate = line[0][5]
                        ProStopDate = line[0][6]
                        if ProStopDate == future_market_date(ProRunDate, interval, conn):
                            # print(ProRunDate, ProStopDate, re.sub('ismlist', 'ismfactors', f.group()))
                            file_name_ls.extend([f.group(), re.sub('ismlist', 'ismfactors', f.group())])
                        break
    return file_name_ls
        



























