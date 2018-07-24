# encoding: utf-8
import numpy as np
import scipy as sp
import pandas as pd
import sys
import os
import re
import ism_logger

class filter_gate(object):
    """
    doc is missing ...
    """
    
    def __init__(self):
        pass
    
    def stop_loss_gate(self, output, label):
        """
        doc is missing ...
        """
        assert len(output) == len(label)
        cnt1 = 0
        cnt2 = 0
        for _i, res in enumerate(output):
            if res[0] > res[1]:
                cnt1 += 1
                if label[_i][0] == 1.0:
                    cnt2 += 1
        cnt3 = 0
        for res in label:
            if res[0] == 1.0:
                cnt3 += 1
        df = pd.DataFrame(label)
        df.to_csv('check_df.csv', index=None, columns=None)
        
        return float(cnt3/len(label)), cnt3, float(cnt2/cnt1), cnt2
        
    def winning_gate(self, output, label):
        pass
    
    def filter_output(self, output, ismcode):
        """
        doc is missing ...
        """
        assert len(output) == len(ismcode)
        code_ls = []
        for _i, res in enumerate(output):
            if res[0] > res[1]:
                code_ls.append(ismcode[_i][0])
        
        return np.asarray(code_ls)
                
    













































