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

class SF_map(object):
    """
    generate sparse tensor for each ism
    """
    
    def __init__(self):
        
        # if os.path.isfile('./SF_map.csv'):
        #     self.SF_ls = pd.read_csv('./SF_map.csv')
        #     self.SF_ls['SecuCode'] = self.SF_ls['SecuCode'].astype(object)
        # else:
        #     self.SF_ls = self.SF_map()
        #     self.SF_ls.to_csv('./SF_map.csv', header=True, index=True)
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


class ism_list(object):
    """
    ism_list class which preprocesses input data into training, validation set
    with multi-hot encoding
    """
    
    def __init__(self, ism_path, interval, Rc=0.0, mode=0):
        
        self.ism_path = ism_path
        self.interval = interval
        self.Rc = Rc
        self.mode = mode
        self.col_name = self.get_col_name()
        self.SF_map = SF_map()
        self.SF_ls = self.SF_map.SF_ls
        self.N_logit = len(self.SF_ls)
        
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
        data = np.array(data[1:])    
                
        return data
    
    def get_col_name(self):
        """
        automatically get the column name of ismlist
        """
        return ['ismcode', 'C1', 'C2', 'C3', 'C4']
        
    def ism_logit_and_labels(self, ism_input):
        """
        label all ism data of training set
        ism_input: path of ism file
        待更新: 增加根据文件名进行的日期检查，避免不必要的查询开销
        """
        # preallocate label list
        logit_labels = []
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
                    sparse_ls = []
                    for i in SecuCode_ls:
                        sparse_ls.append(self.SF_ls[self.SF_ls['SecuCode']==i].index.tolist()[0])
                    # give labels and sparse code
                    if res > self.Rc and res > ExpReturn:
                        logit_labels.append([ismcode, 1.0, 0.0, 0.0, 0.0] + sparse_ls) 
                    elif res > self.Rc and res <= ExpReturn:
                        logit_labels.append([ismcode, 0.0, 1.0, 0.0, 0.0] + sparse_ls) 
                    elif res <= self.Rc and res > ExpReturn:
                        logit_labels.append([ismcode, 0.0, 0.0, 1.0, 0.0] + sparse_ls) 
                    elif res <= self.Rc and res <= ExpReturn:
                        logit_labels.append([ismcode, 0.0, 0.0, 0.0, 1.0] + sparse_ls) 
                               
            conn.close()
        # testing mode
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
                # sparse code
                sparse_ls = []
                for i in SecuCode_ls:
                    sparse_ls.append(self.SF_ls[self.SF_ls['SecuCode']==i].index.tolist()[0])
                logit_labels.append([ismcode, 0.0, 0.0, 0.0, 0.0] + sparse_ls)
        
        return logit_labels
    
    def logit_and_labels(self):
        """
        doc is missing ...
        """
        # database address
        jydb_address = "117.122.223.35"
        jydb_user_id = "zpy"
        jydb_user_pwd = "Z1pe1y1@zpy"
        jydb_db_name = "jydb02"
        # on-line mode
        conn = pymysql.connect(jydb_address, jydb_user_id, jydb_user_pwd, jydb_db_name)
        # preallocate
        ls = []
        for file_name in interval_filter(self.ism_path, self.interval, conn):
            f = re.match(r'ismfactors\.(\d{4})-?(\d{2})-?(\d{2}).*', file_name)
            if f is None:
                continue
            else:
                f2 = os.path.join(self.ism_path, re.sub('ismfactors', 'ismlist', f.group()))
                if not os.path.isfile(f2):
                    continue
                else:
                    ls.append(f2)
        # collect logit and labels with multiprocessing
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
            for _, i in tqdm(enumerate(p.imap_unordered(self.ism_logit_and_labels, ls))):
                df_list.append(i)
                pbar.update()
        p.close()
        p.join()
        logit_labels = []
        for i in df_list:
            logit_labels.extend(i)
        
        return logit_labels
        
    def dense_encoding(self, logit_labels_ls):
        """
        dense encoding
        """
        input_batch = []
        label_batch = []
        for logit_labels in logit_labels_ls:
            dense_code = np.zeros((self.N_logit,), dtype=np.float32)
            for i in logit_labels[5:]:
                dense_code[int(i)] = 1.0
            input_batch.append(dense_code)
            label_batch.append(logit_labels[1:5])
        return input_batch, label_batch


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
        flag = os.path.split(ism_input)[-1].split('.')[-2] + \
               '.' + os.path.split(ism_input)[-1].split('.')[-1]
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
        # database address
        jydb_address = "117.122.223.35"
        jydb_user_id = "zpy"
        jydb_user_pwd = "Z1pe1y1@zpy"
        jydb_db_name = "jydb02"
        # on-line mode
        conn = pymysql.connect(jydb_address, jydb_user_id, jydb_user_pwd, jydb_db_name)
        # preallocate
        ls = []
        for file_name in interval_filter(self.ism_path, self.interval, conn):
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
    待更新: factors维度改为自动获取
    """
    
    def __init__(self, ism_path, interval, batch_size, scaler_ls=None, Rc=0.0, mode=0):
        
        self.ism_path = ism_path
        self.batch_size = batch_size
        self.interval = interval
        self.Rc = Rc
        self.mode = mode
        # self.N_feature = 19
        self.ism_list= ism_list(self.ism_path, self.interval, self.Rc, self.mode)
        self.ism_factor = ism_factor(self.ism_path, self.interval, scaler_ls)
        self.data_logit_labels = self.ism_list.logit_and_labels()
        self.data_factor, self.scaler_ls = self.ism_factor.standard_factor()
        assert len(self.data_logit_labels) == len(self.data_factor)
        self.num_data = len(self.data_factor)
        self.N_logit = len(self.ism_list.SF_ls)
        self.N_factor = 19
        # output standard scaler list
        joblib.dump(self.scaler_ls, './standard_scaler_list.pkl')
    
    def to_sparse_indice(self, logit_batch):
        """
        transform logit batch to sparse indices
        """
        indices = []
        for _cnt, i in enumerate(logit_batch):
            ls = i
            ls.extend([_cnt]*len(i))
            ls = np.asarray(ls)
            ls = np.flip(ls.reshape((2,-1)).transpose(), axis=1).tolist()
            indices.extend(ls) 
        
        return indices
    
    def next_batch(self):
        """
        random sample
        待更新: 先把Data进行shuffle，然后mod顺序取，保证遍历性
        注意: 务必保持logit，labels，factor三者具有相同的序
        待更新: 加入可自定义的求和权重(weight)和sparse tensor shape
        """
        # random sampling ref to logit and labels list
        sample_logit_labels = random.sample(self.data_logit_labels, self.batch_size)
        index = list(map(lambda x: x[0], sample_logit_labels))
        label_batch = list(map(lambda x: x[1:5], sample_logit_labels))
        logit_batch = list(map(lambda x: x[5:], sample_logit_labels))
        logit_idsval = flatten(logit_batch)
        logit_shape = [self.batch_size, self.N_logit]
        # sample and reorder factor list
        sample_factor = self.data_factor[self.data_factor['ismcode'].isin(index)].copy()
        sample_factor['ismcode'] = sample_factor['ismcode'].astype('category')
        sample_factor['ismcode'].cat.reorder_categories(index, inplace=True)
        sample_factor.sort_values('ismcode', inplace=True)
        # gathering batch data
        factor_batch = sample_factor.drop(columns='ismcode').values
        
        return self.to_sparse_indice(logit_batch), logit_idsval, logit_shape, \
               factor_batch, label_batch
    
    def validation_set(self):
        all_logit_and_labels = self.data_logit_labels
        index = list(map(lambda x: x[0], all_logit_and_labels))
        all_label = list(map(lambda x: x[1:5], all_logit_and_labels))
        all_logit = list(map(lambda x: x[5:], all_logit_and_labels))
        all_idsval = flatten(all_logit)
        logit_shape = [self.batch_size, self.N_logit]
        # reorder factor list
        all_factor = self.data_factor[self.data_factor['ismcode'].isin(index)].copy()
        all_factor['ismcode'] = all_factor['ismcode'].astype('category')
        all_factor['ismcode'].cat.reorder_categories(index, inplace=True)
        all_factor.sort_values('ismcode', inplace=True)
        all_factor = all_factor.drop(columns='ismcode').values
        
        return self.to_sparse_indice(all_logit), all_idsval, logit_shape, \
               all_factor, all_label
        
    def evaluation_set(self):
        pass





























