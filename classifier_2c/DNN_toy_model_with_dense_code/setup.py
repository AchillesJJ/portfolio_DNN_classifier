# encoding: utf-8
import subprocess as sub
import datetime
import argparse
import pymysql
import time
import sys
import os
import ism_logger
from utility import *

def job1():
    """
    下载在程序运行当天及之前已到期的所有7天ism数据
    """
    # download history data
    sub.call("python data_downloader.py -mode 1", shell=True)

def job2():
    """
    将历史数据分发到训练集文件夹和测试集文件夹    
    并使用历史数据进行初始测试，判断分类器对近期是否有效
    """
    # manage data
    sub.call("python data_manager.py -mode 1", shell=True)
    # trial model training
    sub.call("python cascading_DNNClassifier.py --num-epoch 41", shell=True)
    # model validation
    sub.call("python cascading_DNNClassifier.py -mode 1", shell=True)

def job3():
    """
    将全部历史数据分发到训练集文件
    使用全部历史数据正式训练分类器模型
    """
    # manage data 
    sub.call("python data_manager.py -mode 2", shell=True)
    # true model training
    sub.call("python cascading_DNNClassifier.py --num-epoch 41", shell=True)

def job4():
    """
    下载指定日期的或最新待过滤的ism数据，并放入对应文件夹
    """ 
    # download newest data
    sub.call("python data_downloader.py -mode 2", shell=True)
    sub.call("python data_manager.py -mode 3", shell=True)

def job5():
    """
    通过多指标综合排序选择最优模型用于过滤输出
    """
    # true model evaluation
    sub.call("python cascading_DNNClassifier.py -mode 2", shell=True)

def job6():
    """
    上传过滤ism到指定节点文件夹
    """
    # data upload
    sub.call("python data_uploader.py", shell=True)


def main(args):
    """
    doc is missing ...
    """
    # connect database
    # jydb_address = "10.1.10.101"
    # jydb_user_id = "zpy"
    # jydb_user_pwd = "Z1pe1y1@zpy"
    # jydb_db_name = "jydb02"
    # conn = pymysql.connect(jydb_address, jydb_user_id, jydb_user_pwd, jydb_db_name)
    
    # on line
    while True:
        time.sleep(10)
        localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        localtime = localtime.split(' ')
        # 每天下午5点开始下载历史数据进行预训练
        if isInTime(localtime[1], '17:00:00', '17:05:00'):
            # 判断当天是否交易日
            # connect database
            jydb_address = "10.1.10.101"
            jydb_user_id = "zpy"
            jydb_user_pwd = "Z1pe1y1@zpy"
            jydb_db_name = "jydb02"
            conn = pymysql.connect(jydb_address, jydb_user_id, jydb_user_pwd, jydb_db_name)
            if is_market_day(localtime[0], conn):
                job1()
                job2()
                job3()
            time.sleep(360)
            conn.close()
        # 第二天凌晨2点开始下载待过滤的ism
        if isInTime(localtime[1], '02:00:00', '02:05:00'):
            # 判断当天是否交易日
            # connect database
            jydb_ddress = "10.1.10.101" 
            jydb_user_id = "zpy"
            jydb_user_pwd = "Z1pe1y1@zpy"
            jydb_db_name = "jydb02"
            conn = pymysql.connect(jydb_address, jydb_user_id, jydb_user_pwd, jydb_db_name)
            if is_market_day(localtime[0], conn):
                job4()
                job5()
                job6()
            time.sleep(360)
            conn.close()
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-nd', '--new-date', help='date of newest ism', default=None)
    args = parser.parse_args()
    
    # run main program    
    main(args)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


