# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 15:49:24 2022

@author: 范朝
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
from scipy.optimize import curve_fit
mpl.rcParams.update(mpl.rcParamsDefault)
df = pd.read_csv('td 5% (1).csv')
B_mass = df['B0_MM']
heights,edges,_ = plt.hist(B_mass,bins=200,range=[5180,5700],label='histogram')
#plt.rcParams['text.usetex'] = True
plt.xlabel('B_0 mass')
plt.ylabel('Number')

#%%
def bckgrd(x,a,b):
    return a*np.exp(-b*(x-5200))

def gaussian(x,sigma,A,mu):
    return A*np.exp(-(x-mu)**2/sigma**2)

def combined(x,a,b,sigma,A,mu):
    return bckgrd(x,a,b) + gaussian(x,sigma,A,mu)

#%%
p0 = [1500,0.0001,50,2000,5300]
x = edges[:-1]

popt, pcov = curve_fit(combined, x, heights,p0=p0)
plt.plot(x,combined(x,*popt),label='fit')
plt.legend()
plt.show()
#%%
#plt.plot(x,heights)