# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 15:34:32 2022

@author: jingyi
"""
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from iminuit import Minuit
import glob
#%%
os.chdir(r'C:\Users\jingyi\OneDrive - Imperial College London\Desktop\year3-problem-solving')
#cwd=os.getcwd()
#%%  
# specifying the path to csv files
path = r"C:\Users\jingyi\OneDrive - Imperial College London\Desktop\year3-problem-solving"
  
# csv files in the path
files = glob.glob(path + "/*.csv")
  
# assign dataset names
list_of_names = ['acceptance_mc','jpsi']
 
# create empty list
dataframes_list = []
 
# append datasets into teh list
for i in range(len(list_of_names)):
    temp_df = pd.read_csv(path+"/"+list_of_names[i]+".csv")
    dataframes_list.append(temp_df)
#ref:https://www.geeksforgeeks.org/read-multiple-csv-files-into-separate-dataframes-in-python/
#%%
td=pd.read_csv('total_dataset.csv')
sig=pd.read_csv('acceptance_mc.csv')
#pkmm=pd.read_csv('k_pi_swap.csv')
#%%
MM=td['B0_MM']
q2=td['q2']
q2s=sig['q2']
#thetal_s=sig['costhetal']
#pkmmMM=pkmm['B0_MM']
#%%
plt.hist(MM,bins=1000)
plt.xlabel('B0_MM/MeV')
plt.ylabel('Number of candiates')
plt.title('Rough examination(no filtering applied')
#%%
plt.hist
#%%distribution in q2
plt.hist(q2s,bins=1000)
plt.xlabel('q2/GeV')
plt.ylabel('Number of candiates')
plt.title('q2 sample distribution(no filtering applied')
#%%
mfilter= td[(td.B0_MM <=5866)&(td.B0_MM>4779)]