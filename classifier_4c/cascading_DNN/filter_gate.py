# encoding: utf-8
import numpy as np
import scipy as sp
import pandas as pd
import sys
import os
import re

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
            if np.argmax(res) in [0, 1]:
                cnt1 += 1
                if label[_i][0]==1.0 or label[_i][1]==1.0:
                    cnt2 += 1
        cnt3 = 0
        for res in label:
            if res[0]==1.0 or res[1]==1.0:
                cnt3 += 1
        
        return float(cnt3/len(label)), float(cnt2/cnt1)
        
    def winning_gate(self, output, label):
        pass