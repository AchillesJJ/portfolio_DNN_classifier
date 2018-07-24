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
from sklearn.externals import joblib
import requests
import time
import multiprocessing


class ism_list(object):
    """
    ism_list class which preprocesses input data into training, validation set
    with one-hot encoding
    """
    
    def __init__(self, ism_path, interval, Rc=0.0, mode=0):
        
        self.ism_path = ism_path
        self.interval = interval
        self.Rc = Rc
        self.mode = mode
        self.col_name = self.get_col_name()
        
    def structure(self, ism_input):
        """
        read in ism_input ASCII text file and transforms to structured data
        ism_input: string, file path of ism file
        return: a structured data set (numpy list)
        """
        f = open(ism_input, 'r')
        flag = '.'.join(os.path.split(ism_input)[-1].split('.')[-4:])
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
        data = np.array(data[1:])    
                
        return data
    
    def get_col_name(self):
        """
        automatically get the column name of ismlist
        """
        return ['ismcode', 'C1', 'C2']
        
    def ism_label(self, ism_input):
        """
        label all ism data of training set
        ism_input: path of ism file
        待更新: 增加根据文件名进行的日期检查，避免不必要的查询开销
        """
        # preallocate label list
        labels = []
        # training/validation mode
        if self.mode == 0 or self.mode == 1:
            # database address
            jydb_address = "10.1.10.101"
            jydb_user_id = "zpy"
            jydb_user_pwd = "Z1pe1y1@zpy"
            jydb_db_name = "jydb02"
            # preallocate real statistics dict
            labels = []
            # on-line mode
            conn = pymysql.connect(jydb_address, jydb_user_id, jydb_user_pwd, jydb_db_name)
            # iterate on all ism packages
            data = self.structure(ism_input)
            # (1) collect common info
            if len(data): # not empty
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
                    df = pd.merge(run_price, stop_price, on='InnerCode', suffixes=('_run', '_stop'))
                    # handle data missing
                    df = df.replace(0.0, np.nan)
                    df = df.dropna(axis=0, how='any')
                    
                    if df.empty:
                        res = np.nan
                        continue
                    else:
                        run_price = df['ClosePrice_run'].sum()
                        stop_price = df['ClosePrice_stop'].sum()
                        if run_price > 0 and stop_price > 0:
                            res = ((stop_price-run_price)/run_price)*250/interval # 243 market days/year
                            # give labels
                            if res > self.Rc:
                                labels.append([ismcode, 1.0, 0.0]) 
                            else:
                                labels.append([ismcode, 0.0, 1.0]) 
                        else:
                            continue        
                labels = pd.DataFrame(labels, columns=['ismcode', 'C1', 'C2'])                           
            conn.close()
        # test mode
        else:
            # iterate on all ism packages
            data = self.structure(ism_input)
            # (1) collect common info
            if len(data):
                pack = data[0]
                interval = float(pack[0][-5])
                ProRunDate = pack[0][5]
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
                    labels.append([ismcode, 0.0, 0.0])
                labels = pd.DataFrame(labels, columns=['ismcode', 'C1', 'C2'])   
                    
        return labels
    
    def labels(self):
        """
        doc is missing ...
        """
        # preallocate
        ls = []
        for file_name in interval_filter(self.ism_path, self.interval):
            f = re.match(r'ismfactors\.(\d{4})-?(\d{2})-?(\d{2}).*', file_name)
            if f is None:
                continue
            else:
                f2 = os.path.join(self.ism_path, re.sub('ismfactors', 'ismlist', f.group()))
                if not os.path.isfile(f2):
                    continue
                else:
                    ls.append(f2)
        # collect labels with multiprocessing
        # (1) define num of preocessings
        if len(ls) < multiprocessing.cpu_count():
            n = len(ls)
        else:
            n = multiprocessing.cpu_count()
        # (2) Pool map
        p = multiprocessing.Pool(n)
        # df_list = p.map(self.ism_label, ls)
        df_list = []
        with tqdm(total=len(ls)) as pbar:
            for _, i in tqdm(enumerate(p.imap_unordered(self.ism_label, ls))):
                df_list.append(i)
                pbar.update()
        p.close()
        p.join()
        df_list = pd.concat(df_list, ignore_index=True)
        return df_list


class ism_factor(object):
    """
    prepare ism factors
    """
    
    def __init__(self, ism_path, interval, scaler_ls=None):
        
        self.ism_path = ism_path
        self.interval = interval
        self.scaler_ls = scaler_ls
        self.col_name = self.get_col_name()
        
    def get_col_name(self):
        """
        automatically get the column name of ismlist
        """
        for file_name in os.listdir(self.ism_path):
            f = re.match(r'ismfactors\.(\d{4})-?(\d{2})-?(\d{2}).*', file_name)
            if f is None:
                continue
            else:
                f2 = open(os.path.join(self.ism_path, f.group()), 'r')
                line = f2.readline()
                line = line.replace('\"', '')
                line = line.replace('[', '')
                line = line.replace(']', '')
                line = line.strip()
                line = line.split('|')
                col_name = [line[0]] + line[5:]
                break
        return col_name
    
    def structure(self, ism_input):
        """
        read in ism factor ASCII file and transforms to structured data (dataframe)
        """
        f = open(ism_input, 'r')
        flag = '.'.join(os.path.split(ism_input)[-1].split('.')[-4:])
        cnt = 0
        for line in f:
            line = line.replace('\"', '')
            line = line.replace('[', '')
            line = line.replace(']', '')
            line = line.strip()
            line = line.split('|')
            if cnt > 0:
                line[0] = line[0] + '.' + flag
                data = pd.DataFrame([[line[0]] + line[5:]], columns=col_name)
                df = df.append(data, ignore_index=True)
            else:
                col_name = [line[0]] + line[5:]
                df = pd.DataFrame(columns=col_name)
            cnt += 1
        # change type
        for col in col_name[1:]:
            df[col] = df[col].astype(float)        
        return df
    
    def standard_factor(self):
        """
        normalize all factors in ism_factor and return a dict
        """
        # preallocate
        ls = []
        for file_name in interval_filter(self.ism_path, self.interval):
            f = re.match(r'ismfactors\.(\d{4})-?(\d{2})-?(\d{2}).*', file_name)
            if f is None:
                continue
            else:
                f2 = os.path.join(self.ism_path, re.sub('ismfactors', 'ismlist', f.group()))
                if not os.path.isfile(f2):
                    continue
                else:
                    ls.append(os.path.join(self.ism_path, f.group()))
        # collect factors with multiprocessing
        # (1) define num of preocessings
        if len(ls) < multiprocessing.cpu_count():
            n = len(ls)
        else:
            n = multiprocessing.cpu_count()
        # (2) Pool map
        p = multiprocessing.Pool(n)
        # df_factor = p.map(self.structure, ls)
        df_factor = []
        with tqdm(total=len(ls)) as pbar:
            for _, i in tqdm(enumerate(p.imap_unordered(self.structure, ls))):
                df_factor.append(i)
                pbar.update()
        p.close()
        p.join()
        df_factor = pd.concat(df_factor, ignore_index=True)
                 
        # sklearn standard scaler
        if self.scaler_ls is None:
            scaler_ls = {}
            for col_name in df_factor.columns.values.tolist()[1:]: # no 'ismcode' column
                ls = np.reshape(df_factor[col_name].values, (-1, 1))
                scaler_ls[col_name] = preprocessing.StandardScaler().fit(ls)
                df_factor[col_name] = np.reshape(scaler_ls[col_name].transform(ls), (-1))
            return df_factor, scaler_ls
        else:
            for col_name in df_factor.columns.values.tolist()[1:]: # no 'ismcode' column
                ls = np.reshape(df_factor[col_name].values, (-1, 1))
                df_factor[col_name] = np.reshape(self.scaler_ls[col_name].transform(ls), (-1))
            return df_factor, self.scaler_ls


class ism_data(object):
    """
    combine ism_factors and ism_list
    待更新: 增加本地文件模式 i.e. data is not None
    待更新: 处理list和factor合并后产生的缺省
    """
    
    def __init__(self, ism_path, batch_size, interval=7, scaler_ls=None, Rc=0.0, mode=0):
        
        self.ism_path = ism_path
        self.interval = interval
        self.batch_size = batch_size
        self.Rc = Rc
        self.mode = mode
        self.N_feature = 19
        self.ism_list= ism_list(self.ism_path, self.interval, self.Rc, self.mode)
        self.ism_factor = ism_factor(self.ism_path, self.interval, scaler_ls)
        self.std_factor, self.scaler_ls = self.ism_factor.standard_factor()
        self.data = self.combine(self.ism_list.labels(), self.std_factor)
        self.num_data = len(self.data)
        # output standard scaler list
        joblib.dump(self.scaler_ls, './standard_scaler_list.pkl')
        print("logit and factor num is {}".format(self.num_data))
    
    def combine(self, df1, df2):
        """
        merge two dataframe
        """
        return pd.merge(df1, df2, on='ismcode')
    
    def next_batch(self, batch_size=None):
        if batch_size is None:
            sample = self.data.sample(n=self.batch_size)
        else:
            sample = self.data.sample(n=batch_size)
        label_batch = sample[['C1','C2']].values
        sample = sample.drop(columns=['ismcode','C1','C2'])
        input_batch = sample.values
        return input_batch, label_batch
    
    def validation_set(self):
        labels = self.data[['C1','C2']].values
        inputs = self.data.drop(columns=['ismcode','C1','C2'])
        inputs = inputs.values
        return inputs, labels
    
    def evaluation_set(self):
        ismcode = self.data['ismcode'].values
        ismcode = np.asarray(list(map(lambda x: [x], ismcode)))
        inputs = self.data.drop(columns=['ismcode','C1','C2']).values
        return inputs, ismcode
        
        
        