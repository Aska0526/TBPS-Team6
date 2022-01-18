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
#%%
os.chdir(r'C:\Users\jingyi\OneDrive - Imperial College London\Desktop\year3-problem-solving')
#cwd=os.getcwd()
#print("Current working directory: {0}".format(cwd))
#%%
df = pd.read_csv('psi2S.csv')
total=pd.read_csv('total_dataset.csv')
#%%
MM=df['Kstar_MM']
plt.hist(MM,bins=40)
#%%
