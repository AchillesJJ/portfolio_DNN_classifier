# encoding: utf-8
import subprocess as sub
import datetime
import pymysql
import argparse
import time
import os
from utility import *
import ism_logger
import logging

class importer(object):
    
    def __init__(self, IP, user, cd_path, lcd_path, nip_path, interval):
        """
        doc is missing ...
        """
        # database address
        jydb_address = "10.1.10.101"
        jydb_user_id = "zpy"
        jydb_user_pwd = "Z1pe1y1@zpy"
        jydb_db_name = "jydb02"
        # on-line mode
        self.conn = pymysql.connect(jydb_address, jydb_user_id, jydb_user_pwd, jydb_db_name)
        self.IP = IP
        self.user = user
        self.cp = cd_path
        self.lcp = lcd_path
        self.nip = nip_path
        self.interval = interval
        self.today = datetime_to_string(datetime.date.today())
        self.num_hist = 6 # num of days for history data
        self.begin_date, self.end_date = self.date_range()
        self.logger = logging.getLogger('ism_logger')
    
    def date_range(self):
        """
        doc is missing ...
        """
        if is_market_day(self.today, self.conn):
            end_date = former_market_date(self.today, self.interval, self.conn)
        else:
            end_date = former_market_date(self.today, self.interval+1, self.conn)
        begin_date = former_market_date(end_date, self.num_hist, self.conn)
        # change to ism generation day (1 day earlier)
        begin_date = string_to_datetime(begin_date)
        begin_date -= datetime.timedelta(days=1)
        end_date = string_to_datetime(end_date)
        end_date -= datetime.timedelta(days=1)
        
        return begin_date, end_date
        
    def hist_data_download(self):
        """
        doc is missing ...
        """
        # download history data
        date = self.begin_date
        while date <= self.end_date:
            # check whether directory exists
            pass
            # download
            file_name = "".join(datetime_to_string(date).split('-'))
            flag1 = "scp -r " + self.user + "@" + self.IP + ":" + \
                    os.path.join(self.cp, file_name, '*ismlist.[0-9]*.[0-9]*.' + \
                                 str(self.interval) + '.[0-9]*.[0-9]* ') + self.lcp
            flag2 = "scp -r " + self.user + "@" + self.IP + ":" + \
                    os.path.join(self.cp, file_name, '*ismfactors.[0-9]*.[0-9]*.' + \
                                 str(self.interval) + '.[0-9]*.[0-9]* ') + self.lcp
            sub.call(flag1, shell=True)
            sub.call(flag2, shell=True)
            date += datetime.timedelta(days=1)
        
        # delete extra space and other character
        for file_name in os.listdir(self.lcp):
            new_name = file_name.strip()
            os.rename(os.path.join(self.lcp, file_name), os.path.join(self.lcp, new_name))
            
    def new_data_download(self):
        """
        doc is missing ...
        """     
        new_ism_date = former_market_date(self.today, 2, self.conn) 
        file_name = ''.join(new_ism_date.split('-'))
        flag1 = "scp -r " + self.user + "@" + self.IP + ":" + \
                os.path.join(self.cp, file_name, '*ismlist.[0-9]*.[0-9]*.' + \
                             str(self.interval) + '.[0-9]*.[0-9]* ') + self.nip
        flag2 = "scp -r " + self.user + "@" + self.IP + ":" + \
                os.path.join(self.cp, file_name, '*ismfactors.[0-9]*.[0-9]*.' + \
                             str(self.interval) + '.[0-9]*.[0-9]* ') + self.nip
        sub.call(flag1, shell=True)
        sub.call(flag2, shell=True)
        # delete extra space and other character
        for file_name in os.listdir(self.nip):
            new_name = file_name.strip()
            os.rename(os.path.join(self.nip, file_name), os.path.join(self.nip, new_name))
        
        return new_ism_date

def main(args):
    
    # logger
    logger = logging.getLogger('ism_logger')
    
    # importer
    imp = importer(IP=args.IP, user=args.user, cd_path=args.cd_path,
                   lcd_path=args.local_cd_path, nip_path=args.new_ism_path,
                   interval=args.interval)
    if args.mode == 1: # download hist data
        # check data file exists or not
        if not os.path.exists(args.local_cd_path):
            os.mkdir(args.local_cd_path)
        else:
            del_file(args.local_cd_path)
        print("history data ranges from {} to {}". \
              format(datetime_to_string(imp.begin_date), datetime_to_string(imp.end_date)))
        imp.hist_data_download()
        logger.info('history data ranging from {} to {} is downloaded.'. \
                    format(datetime_to_string(imp.begin_date), datetime_to_string(imp.end_date)))
    elif args.mode == 2: # download new data
        # check data file exists or not
        if not os.path.exists(args.new_ism_path):
            os.mkdir(args.new_ism_path)
        else:
            del_file(args.new_ism_path)
        
        new_ism_date = imp.new_data_download()
        print('new data is generated at {}'.format(new_ism_date))
        logger.info('new ism data generated at {} is downloaded.'. \
                    format(new_ism_date))
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-IP', help='IP address', default='10.1.14.2')
    parser.add_argument('-user', help='user name', default='root')
    parser.add_argument('-cp', '--cd-path', help='remote path', default='/data/ismcom')
    parser.add_argument('-lcp', '--local-cd-path', help='local path', default='./hist_data')
    parser.add_argument('-nip', '--new-ism-path', help='new ism data path', default='./new_ism')
    parser.add_argument('-mode', help='hist or new data mode', type=int, choices=[1,2])
    parser.add_argument('-nd', '--new-date', help='date of newest ism', default=None)
    parser.add_argument('-interval', help='ism slice cycle', type=int, default=7)
    args = parser.parse_args()
    
    # run main program
    main(args)








