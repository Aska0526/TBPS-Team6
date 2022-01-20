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
# specifying the path to csv files
path = r"pathname"
  
# csv files in the path
files = glob.glob(path + "/*.csv")
  
# assign dataset names
list_of_names = ['acceptance_mc','jpsi'] #names should be in the order as that in the file
 
# create empty list
dataframes_list = []
 
# append datasets into teh list
for i in range(len(list_of_names)):
    temp_df = pd.read_csv(path+"/"+list_of_names[i]+".csv")
    dataframes_list.append(temp_df)
#ref:https://www.geeksforgeeks.org/read-multiple-csv-files-into-separate-dataframes-in-python/


#assign values
B0MM


#%% rough plot first for any varaible interested in
B0_MM=td['B0_MM']
plt.hist(MM,bins=1000)
plt.xlabel('B0_MM/MeV')
plt.ylabel('Number of candiates')
plt.title('Rough examination(no filtering applied')

#%%

#%% Invariant mass filtering 
BL= #B0_lower_bound~B0 rest mass
BU= #B0_upper_bound
Bfilter= td[(td.B0_MM <=BU)&(td.B0_MM>=BL)] #filters out unwanted data
Bremoved=td[(td.B0_MM >=BU)&(td.B0_MM<=BL)] #these are the removed data for background analysis

#%%
#Other dicussed criterions: PT of products should be large; B0_IPCHI2_OWNPV shouldn't be too big; B0_DIRA~1; all X^2 can't really be too large for the data to make sense; 
#identifying background contributions 
#then finally apply these filtering to the sample data to find acceptance
#filtering order- strongest to weakest restriction (peel off layer by layer)?

