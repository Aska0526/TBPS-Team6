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


#%%
"""
Defines the combinatorial background
and loads the data
"""
def bckgrd(x, a, b):
    return a * np.exp(-b * (x - 5200))


def gaussian(x, sigma, A, mu):
    return A * np.exp(-(x - mu) ** 2 / sigma ** 2)


def combined(x, a, b, sigma, A, mu):
    return bckgrd(x, a, b) + gaussian(x, sigma, A, mu)


df_5 = pd.read_csv(r'chi^2 filtered data\td 5% (1).csv')
df_10 = pd.read_csv(r'chi^2 filtered data\td 10%.csv')
df_30 = pd.read_csv(r'chi^2 filtered data\td 30%.csv')
B_mass = np.array([df_5['B0_MM'], df_10['B0_MM'], df_30['B0_MM']], dtype=object)

#%%
"""
Setup the axes for histogram 
and perform curve fit
"""
f, axs = plt.subplots(3, 1)
heights_5, edges, _ = axs[0].hist(B_mass[0], bins=200, range=[5180, 5700], label='data')
heights_10, edges, _ = axs[1].hist(B_mass[1], bins=200, range=[5180, 5700], label='data')
heights_30, edges, _ = axs[2].hist(B_mass[2], bins=200, range=[5180, 5700], label='data')
# plt.rcParams['text.usetex'] = True
axs[0].set_title('5% sig level')
axs[1].set_title('10% sig level')
axs[2].set_title('30% sig level')
f.supxlabel(r'B0 mass $\left(\frac{MeV}{c^2}\right)$')
f.supylabel('Number')
f.tight_layout()

p0 = [1500, 0.0001, 50, 2000, 5300]
x = np.array([(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)])
popt_5, pcov_5 = curve_fit(combined, x, heights_5, p0=p0)
popt_10, pcov_10 = curve_fit(combined, x, heights_10, p0=p0)
popt_30, pcov_30 = curve_fit(combined, x, heights_30, p0=p0)
axs[0].plot(x, combined(x, *popt_5), label='fit')
axs[0].legend()
axs[1].plot(x, combined(x, *popt_10), label='fit')
axs[2].plot(x, combined(x, *popt_30), label='fit')
plt.show()

