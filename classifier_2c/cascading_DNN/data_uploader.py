import numpy as np
import pandas as pd
import multiprocessing
from utility import *
import datetime
import sys
import os
import re
import subprocess as sub
import linecache
import argparse
import ism_logger
import logging

class exporter(object):
    """
    doc is missing ... 
    """
    
    def __init__(self, IP, user, cd_path, lcd_path, lf_path):
        """
        doc is missing ...
        """
        self.IP = IP
        self.user = user
        self.cp = cd_path
        self.lcp = lcd_path
        self.lfp = lf_path
        self.date = self.get_date()
        self.ismlist_field, self.ismfactors_field = self.get_field_name()
        self.logger = logging.getLogger('ism_logger')
        
    def get_date(self):
        """
        doc is missing ...
        """
        for file_name in os.listdir(self.lcp):
            f = re.match(r'ismlist\.(\d{4})-?(\d{2})-?(\d{2}).*', file_name)
            if f is None:
                continue
            else:
                return f.group(1)+f.group(2)+f.group(3)
                break
    
    def get_field_name(self):
        """
        doc is missing ...
        """
        cnt = 0
        for file_name in os.listdir(self.lcp):
            f = re.match(r'([a-z]+)\.(\d{4})-?(\d{2})-?(\d{2}).*', file_name)
            if f is None:
                continue
            else:
                if f.group(1) == 'ismlist':
                    cnt += 1
                    ismlist_field = linecache.getline(os.path.join(self.lcp, f.group()), 1)
                elif f.group(1) == 'ismfactors':
                    cnt += 1
                    ismfactors_field = linecache.getline(os.path.join(self.lcp, f.group()), 1)
            if cnt == 2:
                break
        
        return ismlist_field, ismfactors_field
    
    def ismcode_structure(self, ismcode):
        """
        doc is missing ...
        """
        ls = ismcode.split('.')
        ls = [int(ls[0][6:])+1, ls[0], self.date + '.' + '.'.join(ls[1:])]
        return ls
    
    def ismlist_structure(self, line):
        """
        doc is missing ...
        """
        pass
        
    def ismfactors_structure(self, line):
        """
        doc is missing ...
        """
        pass
        
    def get_ism(self):
        """
        doc is missing ...
        """
        f1 = np.loadtxt(os.path.join(self.lfp, 'filter_ismcode.txt'), dtype='str')
        # group list by key value
        # (1) collect file name
        fn_ls = []
        for line in f1:
            line = self.ismcode_structure(line)
            fn_ls.append(line[-1])
        fn_ls = list(set(fn_ls))
        # (2) allocate empty dict
        ism_dict = {}    
        for fn in fn_ls:
            ism_dict[fn] = []
        # (3) collect ismcode into dict
        for line in f1:
            line = self.ismcode_structure(line)
            ism_dict[line[-1]].append(line[0:-1])
        
        # get original ism from ismlist and ismfactor
        f2 = open(os.path.join(self.lfp, 'ismlist.'+self.date), 'w')
        f3 = open(os.path.join(self.lfp, 'ismfactors.'+self.date), 'w')
        # add field name to the 1-st line
        f2.write(self.ismlist_field)
        f3.write(self.ismfactors_field)
        cnt = 0
        for key in ism_dict.keys():
            ismlist_path = os.path.join(self.lcp, 'ismlist.'+key)
            ismfactors_path = os.path.join(self.lcp, 'ismfactors.'+key)
            for _i, ismcode in enumerate(ism_dict[key]):
                ism = linecache.getline(ismlist_path, ismcode[0])
                factor = linecache.getline(ismfactors_path, ismcode[0])
                assert ismcode[1] == ism[3:18]
                assert ismcode[1] == factor[2:17]
                # reorder ismcode
                cnt += 1
                ism = ism[0:9] + str(cnt).zfill(9) + ism[18:]
                factor = factor[0:8] + str(cnt).zfill(9) + factor[17:]
                # write to files
                f2.write(ism)
                f3.write(factor)
        
        f2.close()
        f3.close()        
    
    def data_upload(self):
        
        flag1 = "ssh " + self.user + "@" + self.IP + " mkdir " + \
                os.path.join(self.cp, self.date)
        flag2 = "scp -r " + os.path.join(self.lfp, '*ismlist.[0-9]*') + " " + self.user + \
                "@" + self.IP + ":" + self.cp
        flag3 = "scp -r " + os.path.join(self.lfp, '*ismfactors.[0-9]*') + " " + self.user + \
                "@" + self.IP + ":" + self.cp
                
        sub.call(flag1, shell=True)
        sub.call(flag2, shell=True)
        sub.call(flag3, shell=True)
    
def main(args):
    
    # logger
    logger = logging.getLogger('ism_logger')
    
    IP = args.IP
    user = args.user
    cd_path = args.cd_path
    lcd_path = args.local_cd_path
    lf_path = args.local_filter_path
    # data exporter
    exp = exporter(IP, user, cd_path, lcd_path, lf_path)
    exp.get_ism()
    exp.data_upload()
    logger.info('filter ism data is uploaded.')
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-IP', help='IP address', default='10.1.14.2')
    parser.add_argument('-user', help='user name', default='root')
    parser.add_argument('-cp', '--cd-path', help='path to upload data', default='/data/filter_ismcom')
    parser.add_argument('-lcp', '--local-cd-path', help='path to evaluation set', default='./ismdatabak_evaluation')
    parser.add_argument('-lfp', '--local-filter-path', help='path to local filter output', default='./filter_output')
    args = parser.parse_args()
    
    # run main program
    main(args)






























        
    
