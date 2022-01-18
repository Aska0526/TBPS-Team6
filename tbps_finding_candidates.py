# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 15:34:32 2022
@author: jingyi
Finding candidates 
"""
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from iminuit import Minuit
#%%opening up the file 
os.chdir('your local file location')
td=pd.read_csv('total_dataset.csv')

#%% rough plot first for any varaible interested in
B0_MM=td['B0_MM']
plt.hist(MM,bins=1000)
plt.xlabel('B0_MM/MeV')
plt.ylabel('Number of candiates')
plt.title('Rough examination(no filtering applied')

#%% Invariant mass filtering 
BL= #B0_lower_bound~B0 rest mass
BU= #B0_upper_bound
Bfilter= td[(td.B0_MM <BU)&(td.B0_MM>BL)] #filters out unwanted data

#%%Others: PT of products should be large, B 
