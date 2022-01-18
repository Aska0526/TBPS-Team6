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

#%%filtering order- strongest to weakest restriction (peel off layer by layer)

#%% Invariant mass filtering 
BL= #B0_lower_bound~B0 rest mass
BU= #B0_upper_bound
Bfilter= td[(td.B0_MM <=BU)&(td.B0_MM>=BL)] #filters out unwanted data
Bremoved=td[(td.B0_MM >=BU)&(td.B0_MM<=BL)] #these are the removed data for background analysis

#%%
#Other dicussed criterions: PT of products should be large; B0_IPCHI2_OWNPV shouldn't be too big; B0_DIRA~1; all X^2 can't really be too large for the data to make sense; 
#identifying background contributions 
#then finally apply these filtering to the sample data to find acceptance

