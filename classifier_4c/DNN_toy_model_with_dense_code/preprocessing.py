# encoding: utf-8
import tensorflow as tf
import numpy as np
import scipy as sp
import pandas as pd
import pymysql
import random
import cmath
import math
import sys
import os
import re
import argparse
from utility import *
from tqdm import tqdm
from sklearn import preprocessing
import requests
import time
import multiprocessing

class SF_map(object):
    """
    generate sparse tensor for each ism
    """
    
    def __init__(self):
        
        # if os.path.isfile('./SF_map.csv'):
        #     self.SF_ls = pd.read_csv('./SF_map.csv')
        # else:
        #     self.SF_ls = self.SF_map()
        #     self.SF_ls.to_csv('./SF_map.csv', header=True, index=None)
        self.SF_ls = self.SF_map()
        self.N_feature = len(self.SF_ls)
        
    def SF_map(self):
        """
        generate SecuCode-FirstIndustryCode mapping
        """
        # connect database
        jydb_address = "117.122.223.35"
        jydb_user_id = "zpy"
        jydb_user_pwd = "Z1pe1y1@zpy"
        jydb_db_name = "jydb02"
        conn = conn = pymysql.connect(jydb_address, jydb_user_id, jydb_user_pwd, jydb_db_name)
        flag1 = "SELECT CompanyCode, SecuCode FROM  SecuMain WHERE SecuCategory=1 AND SecuMarket IN (83,90)"
        data1 = pd.read_sql(flag1, conn)
        flag2 = "SELECT CompanyCode, FirstIndustryCode FROM LC_ExgIndustry WHERE Standard=3 AND IfPerformed=1 " + \
                "AND CompanyCode IN (" + ",".join(data1['CompanyCode'].astype(str)) + ")"
        data2 = pd.read_sql(flag2, conn)
        data = pd.merge(data1, data2, on='CompanyCode')
        data = data.sort_values(['FirstIndustryCode', 'SecuCode']).\
                reset_index()[['SecuCode', 'FirstIndustryCode']]
        conn.close()
        return data


class ism_data(object):
    """
    ism_data class which preprocesses input data into training, validation set
    with one-hot encoding
    """
    
    def __init__(self, ism_path, interval, Rc=0.0, batch_size=None, mode=0, random_seed=123):
        
        self.ism_path = ism_path
        self.interval = interval
        self.SF_map = SF_map()
        self.SF_ls = self.SF_map.SF_ls
        self.Rc = Rc
        self.N_feature = self.SF_map.N_feature
        self.batch_size = batch_size
        self.mode = mode
        self.sparse_code = self.sparse_encoding()
        self.num_data = len(self.sparse_code)
        # shuffle the sparse code
        random.seed(random_seed)
        # random.shuffle(self.sparse_code)    
        
    def structure(self, ism_input):
        """
        read in ism_input ASCII text file and transforms to structured data
        ism_input: string, file path of ism file
        return: a structured data set (numpy list)
        """
        f = open(ism_input, 'r')
        flag = os.path.split(ism_input)[-1].split('.')[-2] + \
               '.' + os.path.split(ism_input)[-1].split('.')[-1]
        data = []
        cnt = 0
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
            if cnt > 0:
                line[0][0] = line[0][0] + '.' + flag
            line[1:] = ls
            data.append(line)
            cnt += 1
        # exclude the first line
        data = np.array(data[1:])    
                
        return data
        
    def ism_label(self, ism_input):
        """
        label all ism data of training set
        ism_input: path of ism file
        """
        # preallocate label list
        labels = []
        # training/validation mode
        if self.mode == 0 or self.mode == 1:
            # database address
            jydb_address = "117.122.223.35"
            jydb_user_id = "zpy"
            jydb_user_pwd = "Z1pe1y1@zpy"
            jydb_db_name = "jydb02"
            # on-line mode
            conn = pymysql.connect(jydb_address, jydb_user_id, jydb_user_pwd, jydb_db_name)
            # iterate on all ism packages
            data = self.structure(ism_input)
            # (1) collect common info
            pack = data[0]
            interval = float(pack[0][-5])
            ProRunDate = former_market_date(pack[0][5], 2, conn)
            ProStopDate = pack[0][6]
            # (2) iterate on ism portfolio
            for _i, ism_pack in enumerate(data):
                ismcode = ism_pack[0][0]
                MaxStoreSum = ism_pack[0][7]
                MinInvestShare = ism_pack[0][8]
                ExpReturn = float(ism_pack[0][10])
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
                df = pd.merge(run_price, stop_price, on='InnerCode', 
                                                      suffixes=('_run', '_stop'))
                # handle data missing
                df = df.replace(0.0, np.nan)
                df = df.dropna(axis=0, how='any')
                
                if df.empty:
                    res = np.nan
                    continue
                else:
                    run_price = df['ClosePrice_run'].sum()
                    stop_price = df['ClosePrice_stop'].sum()
                    res = ((stop_price-run_price)/run_price)*250/interval # 243 market days/year
                    # sparse_encoding
                    ls = list(map(lambda x: [y[1] for y in x[1:]], ism_pack[1:]))
                    ls = flatten(ls)
                    sparse_ls = []
                    for i in ls:
                        sparse_ls.append(self.SF_ls[self.SF_ls['SecuCode']==i].index.tolist()[0])
                    # give labels
                    if res > self.Rc and res > ExpReturn:
                        labels.append([ismcode, 1.0, 0.0, 0.0, 0.0] + sparse_ls) 
                    elif res > self.Rc and res <= ExpReturn:
                        labels.append([ismcode, 0.0, 1.0, 0.0, 0.0] + sparse_ls) 
                    elif res <= self.Rc and res > ExpReturn:
                        labels.append([ismcode, 0.0, 0.0, 1.0, 0.0] + sparse_ls) 
                    elif res <= self.Rc and res <= ExpReturn:
                        labels.append([ismcode, 0.0, 0.0, 0.0, 1.0] + sparse_ls) 
                               
            conn.close()
        # test mode
        else:
            # iterate on all ism packages
            data = self.structure(ism_input)
            # (1) collect common info
            pack = data[0]
            interval = float(pack[0][-5])
            ProRunDate = former_market_date(pack[0][5], 2, conn)
            ProStopDate = pack[0][6]
            # (2) iterate on ism portfolio
            for _i, ism_pack in enumerate(data):
                ismcode = ism_pack[0][0]
                MaxStoreSum = ism_pack[0][7]
                MinInvestShare = ism_pack[0][8]
                ExpReturn = float(ism_pack[0][10])
                InnerCode_ls = flatten(list(map(lambda x: [y[0] for y in x[1:]], ism_pack[1:])))
                SecuCode_ls = flatten(list(map(lambda x: [y[1] for y in x[1:]], ism_pack[1:])))
                PriorType_ls = flatten(list(map(lambda x: [y[2] for y in x[1:]], ism_pack[1:])))
                # sparse_encoding
                ls = list(map(lambda x: [y[1] for y in x[1:]], ism_pack[1:]))
                ls = flatten(ls)
                sparse_ls = []
                for i in ls:
                    sparse_ls.append(self.SF_ls[self.SF_ls['SecuCode']==i].index.tolist()[0])
                labels.append([ismcode, 0.0, 0.0, 0.0, 0.0] + sparse_ls)
        
        return labels
        
    def sparse_encoding(self):
        """
        sparse tensor representation of ism
        """    
        # database address
        jydb_address = "117.122.223.35"
        jydb_user_id = "zpy"
        jydb_user_pwd = "Z1pe1y1@zpy"
        jydb_db_name = "jydb02"
        # on-line mode
        conn = pymysql.connect(jydb_address, jydb_user_id, jydb_user_pwd, jydb_db_name)
        # gathering all file name
        ls = []
        for file_name in interval_filter(self.ism_path, self.interval, conn):
            f = re.match(r'ismlist\.(\d{4})-?(\d{2})-?(\d{2}).*', file_name)
            if f is None:
                continue
            else:
                f2 = os.path.join(self.ism_path, f.group())
                ls.append(f2)
        # collect all sparse code with labels
        # (1) define num of processing
        if len(ls) < multiprocessing.cpu_count():
            n = len(ls)
        else:
            n = multiprocessing.cpu_count()
        # (2) Pool map
        p = multiprocessing.Pool(n)
        df_list = []
        # df_list = p.map(self.ism_label, ls)
        with tqdm(total=len(ls)) as pbar:
            for _, i in tqdm(enumerate(p.imap_unordered(self.ism_label, ls))):
                df_list.append(i)
                pbar.update()
        p.close()
        p.join()
        sparse_encode = []
        for i in df_list:
            sparse_encode.extend(i)
                    
        return sparse_encode
        
    def dense_encoding(self, sparse_code_ls):
        """
        dense encoding
        """
        input_batch = []
        label_batch = []
        for sparse_code in sparse_code_ls:
            dense_code = np.zeros((self.N_feature,), dtype=np.float32)
            for i in sparse_code[5:]:
                dense_code[int(i)] = 1.0
            # dense_code = dense_code
            input_batch.append(dense_code)
            label_batch.append(sparse_code[1:5])
        return input_batch, label_batch
        
    def next_batch(self):
        """
        give a batch training set
        待更新: 现阶段先采用随机选取batch，后面可能需要修改以保证数据的遍历性
        """
        sparse_batch = random.sample(self.sparse_code, self.batch_size)
        # sparse to dense code
        input_batch, label_batch = self.dense_encoding(sparse_batch)
        return input_batch, label_batch
    
    def validation_set(self):
        """
        give whole validation set of dense code
        """
        input_all, label_all = self.dense_encoding(self.sparse_code)
        return input_all, label_all
    
    def evaluation_set(self):
        """
        give whole evaluation set of dense code
        """
        pass












































