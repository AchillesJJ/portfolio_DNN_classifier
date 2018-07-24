# encoding: utf-8
import numpy as np
import pandas as pd
import scipy as sp
import pymysql
import datetime
import cmath
import math
import sys
import os
import re
from utility import *
from tqdm import tqdm

class ISM(object):
    """
    doc is missing ...
    """
    def __init__(self, ism_input, moving_average=False, L_shift=0, R_shift=0):
    
        self.moving_average = moving_average
        self.L_shift, self.R_shift = L_shift, R_shift
        self.ism_data = self.structure(ism_input)
        self.back_testing_data = self.hist_data()
        self.ism_label = self.back_testing()
    
    
    def structure(self, ism_input):
        """
        read in ism_input ASCII text file and transforms to structured data
        ism_input: string, file path of ism file
        return: a structured data set (numpy list)
        """
        f = open(ism_input, 'r')
        data = []
        for line in f:
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
            data.append(line)
        data = np.array(data[1:])    
                
        return data
                
    
    def hist_data(self):
        """
        collect all history data needed for ism_input
        return: a structured data set
        """
        # database address
        jydb_address = "117.122.223.35"
        jydb_user_id = "zpy"
        jydb_user_pwd = "Z1pe1y1@zpy"
        jydb_db_name = "jydb02"
        # preallocate data set
        back_testing_data = {}
        
        # on-line mode
        conn = pymysql.connect(jydb_address, jydb_user_id, jydb_user_pwd, jydb_db_name)
        # not using moving average
        if not self.moving_average:                
            # iterate on all ism packages
            for _i, ism_pack in enumerate(tqdm(self.ism_data)):
                ismcode = ism_pack[0][0]
                ProRunDate = ism_pack[0][5]
                ProStopDate = ism_pack[0][6]
                MaxStoreSum = ism_pack[0][7]
                MinInvestShare = ism_pack[0][8]
                InnerCode_ls = flatten(list(map(lambda x: [y[0] for y in x[1:]], ism_pack[1:])))
                SecuCode_ls = flatten(list(map(lambda x: [y[1] for y in x[1:]], ism_pack[1:])))
                PriorType_ls = flatten(list(map(lambda x: [y[2] for y in x[1:]], ism_pack[1:])))
                # collect data from source conn
                flag_run = "SELECT InnerCode, OpenPrice, ClosePrice FROM QT_DailyQuote WHERE " + \
                        "InnerCode IN (" + ",".join(InnerCode_ls) + ") AND " + \
                        "TradingDay=\'" + ProRunDate + "\'"
                flag_stop = "SELECT InnerCode, OpenPrice, ClosePrice FROM QT_DailyQuote WHERE " + \
                        "InnerCode IN (" + ",".join(InnerCode_ls) + ") AND " + \
                        "TradingDay=\'" + ProStopDate + "\'"
                run_price = pd.read_sql(flag_run, conn)
                stop_price = pd.read_sql(flag_stop, conn)
                back_testing_data[ismcode] = pd.merge(run_price, stop_price, on='InnerCode', 
                                                      suffixes=('_run', '_stop'))
        else: # using moving average            
            # iterate on all ism packages
            for _i, ism_pack in enumerate(self.ism_data):
                ismcode = ism_pack[0][0]
                ProRunDate = ism_pack[0][5]
                TradingDay_begin = former_market_date(ProRunDate, self.L_shift, conn)
                TradingDay_end = future_market_date(ProRunDate, self.R_shift, conn)
                InnerCode_ls = flatten(list(map(lambda x: [y[0] for y in x[1:]], ism_pack[1:])))
                SecuCode_ls = flatten(list(map(lambda x: [y[1] for y in x[1:]], ism_pack[1:])))
                PriorType_ls = flatten(list(map(lambda x: [y[2] for y in x[1:]], ism_pack[1:])))
                flag = "SELECT InnerCode, TradingDay, OpenPrice, ClosePrice FROM QT_DailyQuote WHERE " + \
                    "InnerCode IN (" + ",".join(InnerCode_ls) + ") AND " + \
                    "TradingDay BETWEEN \'" + TradingDay_begin + "\' AND \'" + TradingDay_end + "\'"
                back_testing_data[ismcode] = pd.read_sql(flag, conn)
        
        # close sql connection
        conn.close()
    
        return back_testing_data    
        
    
    def back_testing(self, func=None):
        """
        back test for all ism in ism_data set
        func: self-defined lambda function
        return: structured output for later analyze
        """
        # preallocate real statistics dict
        ism_label = []
        
        if not self.moving_average:
            # iterate on all ism packages
            for _i, ism_pack in enumerate(self.ism_data):
                ism_code = ism_pack[0][0]
                interval = float(ism_pack[0][-5])
                df = self.back_testing_data[ism_code]
                # using close price only now, check data is/not empty
                if df.empty:
                    res = np.nan
                    continue
                else:
                    run_price = df['ClosePrice_run'].sum()
                    stop_price = df['ClosePrice_stop'].sum()
                    res = ((stop_price-run_price)/run_price)*243/interval # 243 market days/year
                    ism_label.append([ism_code, res])
        
        else: # using moving_average
            # iterate on all ism packages
            # for _i, ism_pack in enumerate(self.ism_data):
            #     ism_code = ism_pack[0][0]
            #     interval = int(ism_pack[0][-5])
            #     df = self.back_testing_data[ism_code]
            # 
            #     # using close price only now
            #     dim = int(len(df)/(L_shift+R_shift-1))    
            #     close_price = np.array_split(df['ClosePrice'], dim)
            #     f1 = lambda x: float(x[-1]-x[0]) # real return for single stock
            #     f2 = lambda x: float(x[0]) # initial value of single stock
            #     if L_shift+R_shift >= interval:
            #         ls1 = np.zeros((L_shift+R_shift-interval,), dtype=np.float32)
            #         ls2 = np.zeros_like(ls1)
            #         for _j, par in enumerate(close_price):
            #             # sum over different stocks in one ism_pack
            #             par_rolling = par.rolling(interval)
            #             ls1 += np.asarray(par_rolling.apply(f1).dropna())
            #             ls2 += np.asarray(par_rolling.apply(f2).dropna())
            #         ism_label[ism_code] = {}
            #         ism_label[ism_code]['real_return'] = (ls1/ls2/interval)
            #     else:
            #         print("L_shift+R_shift < interval")
            pass    
        
        return ism_label
    
    
    def get_hist_data(self):
        return self.back_testing_data
        
    
    def get_ism_data(self):
        return self.ism_data
        
    
    def get_ism_label(self):
        return self.ism_label








































