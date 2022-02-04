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

# mpl.rcParams.update(mpl.rcParamsDefault)

#%%
"""
Define various functions
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


def bin_num(dataset, num):
    if num == 0:
        dataset = dataset[(dataset['q2'] > 0.1) & (dataset['q2'] > 0.98)]
        return dataset
    elif num == 1:
        dataset = dataset[(dataset['q2'] > 1.1) & (dataset['q2'] > 2.5)]
        return dataset
    elif num == 2:
        dataset = dataset[(dataset['q2'] > 2.5) & (dataset['q2'] > 4.0)]
        return dataset
    elif num == 3:
        dataset = dataset[(dataset['q2'] > 4.0) & (dataset['q2'] > 6.0)]
        return dataset
    elif num == 4:
        dataset = dataset[(dataset['q2'] > 6.0) & (dataset['q2'] > 8.0)]
        return dataset
    elif num == 5:
        dataset = dataset[(dataset['q2'] > 15.0) & (dataset['q2'] > 17.0)]
        return dataset
    elif num == 6:
        dataset = dataset[(dataset['q2'] > 17.0) & (dataset['q2'] > 19.0)]
        return dataset
    elif num == 7:
        dataset = dataset[(dataset['q2'] > 11.0) & (dataset['q2'] > 12.5)]
        return dataset
    elif num == 8:
        dataset = dataset[(dataset['q2'] > 1.0) & (dataset['q2'] > 6.0)]
        return dataset
    elif num == 9:
        dataset = dataset[(dataset['q2'] > 15.0) & (dataset['q2'] > 17.0)]
        return dataset


def acceptance_series(ci, cj, cm, cn, ctl, ctk, phi, q2):
    """
    ci blah blah is the summation series coeff. The size of the input array dictates the order of the polynomial
    ctk, ctl... is the argument of the legendre poly
    """
    sum1 = np.polynomial.legendre.Legendre(ci)
    sum2 = np.polynomial.legendre.Legendre(cj)
    sum3 = np.polynomial.legendre.Legendre(cm)
    sum4 = np.polynomial.legendre.Legendre(cn)
    return sum1(ctl) * sum2(ctk) * sum3(phi) * sum4(q2)


td = pd.read_pickle(r'year3-problem-solving\total_dataset.pkl')
df_5 = pd.read_pickle(r'chi^2 filtered data\td 5% (1).pkl')
df_10 = pd.read_pickle(r'chi^2 filtered data\td 10%.pkl')
df_30 = pd.read_pickle(r'chi^2 filtered data\td 30%.pkl')
B_mass = np.array([df_5['B0_MM'], df_10['B0_MM'], df_30['B0_MM']], dtype=object)

bin_1 = bin_num(df_10, 6)
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

#%%
plt.hist(df_10['costhetal'], bins=200)

#%%
"""
Should (?) show plot the signal noise ratio versus the chi_sq threshold
(whatever that means...)
"""


def full_filter(file, thres, column=None):
    """
    Selects only the likely candidates (maybe?)
    The column arg can be left empty, otherwise can choose the column u want
    """
    file_fild = file[file['Kstar_MM'] > 800]
    file_fild = file_fild[file_fild['B0_IPCHI2_OWNPV'] > thres]
    file_fild = file_fild[file_fild['Pi_IPCHI2_OWNPV'] > thres]
    file_fild = file_fild[file_fild['B0_FDCHI2_OWNPV'] > thres]
    if column is None:
        return file_fild
    else:
        return file_fild[column]


def log_likelihood(thres):
    """
    NOT THE LOG LIKELIHOOD FUNCTION YET...
    This will fit the chopped data and finds the ratio of signal and noise
    """
    p0 = [2000, 0.0003, 5200, 50, 2000, 5300]
    result = []
    for i in thres:
        fil_data = full_filter(td, i, 'B0_MM')
        height, edges, _ = plt.hist(fil_data, bins=200, range=[5180, 5700])
        popt, pcov = curve_fit(combined, edges[:-1], height, p0=p0)
        r = popt[4] / popt[0]
        result.append(r)
    plt.close('all')
    return result


threshold = np.linspace(0.1, 6, 100)

plt.plot(threshold, log_likelihood(threshold), 'x')
plt.show()
