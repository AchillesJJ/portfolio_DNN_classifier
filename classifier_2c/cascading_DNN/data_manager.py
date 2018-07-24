# encoding: utf-8
import subprocess as sub
import datetime
import argparse
import time
import os
from utility import *
import ism_logger
import logging

class manager(object):
    
    def __init__(self, lcd_path, nip_path, mode):
        """
        doc is missing ...
        mode = 1: training and confirming effect
        mode = 2: final training i.e. combine both training and validation set
        """
        self.lcp = lcd_path
        self.nip = nip_path
        self.mode = mode
        self.latest_date = self.get_date()
        self.logger = logging.getLogger('ism_logger')
    
    def get_date(self):
        """
        return the date of validation set
        """
        date_ls = []
        for file_name in os.listdir(self.lcp):
            f = re.match(r'ismlist\.(\d{4})-?(\d{2})-?(\d{2}).*', file_name)
            if f is None:
                continue
            else:
                date_ls.append(string_to_datetime(f.group(1)+'-'+f.group(2)+'-'+f.group(3)))
        latest_date = datetime_to_string(max(date_ls))
        latest_date = latest_date.split('-')
        latest_date = ''.join(latest_date)
        return latest_date
    
    def data_manage(self):
        """
        doc is missing ...
        mode = 1: training and confirming effect
        mode = 2: final training i.e. combine both training and validation set
        """
        if self.mode == 1:
            # check directory and clear obsolete files
            if not os.path.exists('./ismdatabak_training'):
                os.mkdir('./ismdatabak_training')
            else:
                del_file('./ismdatabak_training')
            if not os.path.exists('./ismdatabak_validation'):
                os.mkdir('./ismdatabak_validation')
            else:
                del_file('./ismdatabak_validation')
            # copy files to corresponding directory
            for file_name in os.listdir(self.lcp):
                f = re.match(r'ismlist\.(\d{4})-?(\d{2})-?(\d{2}).*', file_name)
                if f is None:
                    continue
                else:
                    date = f.group(1) + f.group(2) + f.group(3)
                    if date != self.latest_date:
                        flag1 = 'cp ' + os.path.join(self.lcp, f.group()) + ' ./ismdatabak_training'
                        flag2 = 'cp ' + os.path.join(self.lcp, re.sub('ismlist', 'ismfactors', f.group())) + ' ./ismdatabak_training'
                        sub.call(flag1, shell=True)
                        sub.call(flag2, shell=True)
                    else:
                        flag1 = 'cp ' + os.path.join(self.lcp, f.group()) + ' ./ismdatabak_validation'
                        flag2 = 'cp ' + os.path.join(self.lcp, re.sub('ismlist', 'ismfactors', f.group())) + ' ./ismdatabak_validation'
                        sub.call(flag1, shell=True)
                        sub.call(flag2, shell=True)
        elif self.mode == 2:
            # check directory and clear obsolete files
            if not os.path.exists('./ismdatabak_training'):
                os.mkdir('./ismdatabak_training')
            else:
                del_file('./ismdatabak_training')
            # copy files to corresponding directory
            for file_name in os.listdir(self.lcp):
                f = re.match(r'ismlist\.(\d{4})-?(\d{2})-?(\d{2}).*', file_name)
                if f is None:
                    continue
                else:
                    flag1 = 'cp ' + os.path.join(self.lcp, f.group()) + ' ./ismdatabak_training'
                    flag2 = 'cp ' + os.path.join(self.lcp, re.sub('ismlist', 'ismfactors', f.group())) + ' ./ismdatabak_training'
                    sub.call(flag1, shell=True)
                    sub.call(flag2, shell=True)
        elif self.mode == 3:
            # check new ism directory and clear obsolete files
            if not os.path.exists('./ismdatabak_evaluation'):
                os.mkdir('./ismdatabak_evaluation')
            else:
                del_file('./ismdatabak_evaluation')    
            # copy files to corresponding directory
            for file_name in os.listdir(self.nip):
                f = re.match(r'ismlist\.(\d{4})-?(\d{2})-?(\d{2}).*', file_name)
                if f is None:
                    continue
                else:
                    flag1 = 'cp ' + os.path.join(self.nip, f.group()) + ' ./ismdatabak_evaluation'
                    flag2 = 'cp ' + os.path.join(self.nip, re.sub('ismlist', 'ismfactors', f.group())) + ' ./ismdatabak_evaluation'
                    sub.call(flag1, shell=True)
                    sub.call(flag2, shell=True)            
                
def main(args):
    """
    doc is missing ...
    """
    mgr = manager(lcd_path=args.local_cd_path, nip_path=args.new_ism_path, mode=args.mode)
    mgr.data_manage()
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-lcp', '--local-cd-path', help='local path', default='./hist_data')
    parser.add_argument('-nip', '--new-ism-path', help='new ism data path', default='./new_ism')
    parser.add_argument('-mode', help='data manage mode', type=int, choices=[1,2,3])
    args = parser.parse_args()
    
    # run main program
    main(args)































