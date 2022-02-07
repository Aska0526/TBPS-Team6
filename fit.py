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



#%% angular distribution followed by the jupyter notebook
#%% loading data
a_10 = pd.read_csv(open("a_10.csv"))
costhetal_a_10 = a_10["costhetal"]
costhetak_a_10 = a_10["costhetak"]
phi_a_10 = a_10["phi"]

a_30 = pd.read_csv(open("a_30.csv"))
costhetal_a_30 = a_30["costhetal"]
costhetak_a_30 = a_30["costhetak"]
phi_a_30 = a_30["phi"]

td_10 = pd.read_csv(open("td 10%.csv"))
costhetal_td_10 = td_10["costhetal"]
costhetak_td_10 = td_10["costhetak"]
phi_td_10 = td_10["phi"]

td_30 = pd.read_csv(open("td 30%.csv"))
costhetal_td_30 = td_30["costhetal"]
costhetak_td_30 = td_30["costhetak"]
phi_td_30 = td_30["phi"]

#%% histgram
plt.hist(costhetal_a_10, bins = 200, range = [-1,1])
plt.title("histogram of acceptance of $cos(\\theta_l)$")
plt.ylabel("number")
plt.xlabel("$cos(\\theta_l)$")
plt.show()
'''
plt.hist(costhetal_a_30, bins = 200, range = [-1,1])
plt.title("histogram of acceptance of $cos(\\theta_l)$")
plt.ylabel("number")
plt.xlabel("$cos(\\theta_l)$")
plt.show()

plt.hist(costhetal_td_10, bins = 200, range = [-1,1])
plt.title("histogram of acceptance of $cos(\\theta_l)$")
plt.ylabel("number")
plt.xlabel("$cos(\\theta_l)$")
plt.show()

plt.hist(costhetal_td_10, bins = 200, range = [-1,1])
plt.title("histogram of acceptance of $cos(\\theta_l)$")
plt.ylabel("number")
plt.xlabel("$cos(\\theta_l)$")
plt.show()
'''

#%% define fitting function: copy from the jupyter notebook
# only one thing is changed the 0.5 accpetance is replaced by the given data(not sure)

def d2gamma_p_d2q2_dcostheta(fl, afb, cos_theta_l):
    """
    Returns the pdf defined above
    :param fl: f_l observable
    :param afb: a_fb observable
    :param cos_theta_l: cos(theta_l)
    :return:
    """
    ctl = cos_theta_l
    c2tl = 2 * ctl ** 2 - 1
    acceptance = 0.5
    scalar_array = 3/8 * (3/2 - 1/2 * fl + 1/2 * c2tl * (1 - 3 * fl) + 8/3 * afb * ctl) * acceptance
    normalised_scalar_array = scalar_array * 2  # normalising scalar array to account for the non-unity acceptance function
    return normalised_scalar_array

def log_likelihood(fl, afb, test_bin):
    """
    Returns the negative log-likelihood of the pdf defined above
    :param fl: f_l observable
    :param afb: a_fb observable
    :param _bin: number of the bin to fit
    :return:
    """
    ctl = test_bin
    normalised_scalar_array = d2gamma_p_d2q2_dcostheta(fl=fl, afb=afb, cos_theta_l=ctl)
    return - np.sum(np.log(normalised_scalar_array))

#%%
initial_fl = 0
initial_afb = 0.7
x = np.linspace(-1, 1, 500)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
ax1.plot(x, [log_likelihood(fl=initial_fl, afb=initial_afb, test_bin= costhetal_td_10) for i in x])
ax1.set_title(r'$A_{FB}$ = ' + str(initial_afb))
ax1.set_xlabel(r'$F_L$')
ax1.set_ylabel(r'$-\mathcal{L}$')
ax1.grid()
ax2.plot(x, [log_likelihood(fl=initial_fl, afb=initial_afb, test_bin= costhetal_td_10) for i in x])
ax2.set_title(r'$F_{L}$ = ' + str(initial_fl))
ax2.set_xlabel(r'$A_{FB}$')
ax2.set_ylabel(r'$-\mathcal{L}$')
ax2.grid()
plt.tight_layout()
plt.show()
