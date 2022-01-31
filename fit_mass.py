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
from iminuit import Minuit

mpl.rcParams.update(mpl.rcParamsDefault)

#%%
"""
Defines the combinatorial background
and loads the data
"""


def bckgrd(x, a, b, mu_exp):
    """
    The exponentially decaying background
    """
    return a * np.exp(-b * (x - mu_exp))


def gaussian(x, sigma, A, mu_gauss):
    """
    The potential signal
    """
    return A * np.exp(-(x - mu_gauss) ** 2 / sigma ** 2)


def combined(x, a, b, mu_exp, sigma, A, mu_gauss):
    """
    The combined signal pattern
    """
    return bckgrd(x, a, b, mu_exp) + gaussian(x, sigma, A, mu_gauss)


df_5 = pd.read_pickle(r'chi^2 filtered data\td 5% (1).pkl')
df_10 = pd.read_pickle(r'chi^2 filtered data\td 10%.pkl')
df_30 = pd.read_pickle(r'chi^2 filtered data\td 30%.pkl')
B_mass = np.array([df_5['B0_MM'], df_10['B0_MM'], df_30['B0_MM']], dtype=object)

#%%
"""
Setup the axes for histogram 
and perform curve fit
"""
f_B0, axs_B0 = plt.subplots(3, 1)  # Creates a 3 row by 1 column plot
heights_5, edges, _ = axs_B0[0].hist(B_mass[0], bins=200, range=[5180, 5700], label='data')
heights_10, edges, _ = axs_B0[1].hist(B_mass[1], bins=200, range=[5180, 5700], label='data')
heights_30, edges, _ = axs_B0[2].hist(B_mass[2], bins=200, range=[5180, 5700], label='data')
# plt.rcParams['text.usetex'] = True
axs_B0[0].set_title('5% sig level')
axs_B0[1].set_title('10% sig level')
axs_B0[2].set_title('30% sig level')
f_B0.supxlabel(r'B0 mass $\left(\frac{MeV}{c^2}\right)$')  # The overall x label
f_B0.supylabel('Number')
f_B0.tight_layout()  # Prevents the plots from overlapping

p0 = [2100, 0.0003, 5180, 50, 2000, 5300]
x = np.array([(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)])
popt_5, pcov_5 = curve_fit(combined, x, heights_5, p0=p0)
popt_10, pcov_10 = curve_fit(combined, x, heights_10, p0=p0)
popt_30, pcov_30 = curve_fit(combined, x, heights_30, p0=p0)
axs_B0[0].plot(x, combined(x, *popt_5), label='fit')
axs_B0[0].legend()
axs_B0[1].plot(x, combined(x, *popt_10), label='fit')
axs_B0[2].plot(x, combined(x, *popt_30), label='fit')
plt.show()

#%%
"""
Plots histogram of the q^2 distribution (crudely)
"""
q_sq = np.array([df_5['q2'], df_10['q2'], df_30['q2']], dtype=object)
f_q2, axs_q2 = plt.subplots(3, 1)
axs_q2[0].hist(q_sq[0], bins=200)
axs_q2[1].hist(q_sq[1], bins=200)
axs_q2[2].hist(q_sq[2], bins=200)
plt.show()
