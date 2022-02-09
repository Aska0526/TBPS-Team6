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

# Define various functions and loads the data


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
        dataset = dataset[(dataset['q2'] > 0.1) & (dataset['q2'] < 0.98)]
        return dataset
    elif num == 1:
        dataset = dataset[(dataset['q2'] > 1.1) & (dataset['q2'] < 2.5)]
        return dataset
    elif num == 2:
        dataset = dataset[(dataset['q2'] > 2.5) & (dataset['q2'] < 4.0)]
        return dataset
    elif num == 3:
        dataset = dataset[(dataset['q2'] > 4.0) & (dataset['q2'] < 6.0)]
        return dataset
    elif num == 4:
        dataset = dataset[(dataset['q2'] > 6.0) & (dataset['q2'] < 8.0)]
        return dataset
    elif num == 5:
        dataset = dataset[(dataset['q2'] > 15.0) & (dataset['q2'] < 17.0)]
        return dataset
    elif num == 6:
        dataset = dataset[(dataset['q2'] > 17.0) & (dataset['q2'] < 19.0)]
        return dataset
    elif num == 7:
        dataset = dataset[(dataset['q2'] > 11.0) & (dataset['q2'] < 12.5)]
        return dataset
    elif num == 8:
        dataset = dataset[(dataset['q2'] > 1.0) & (dataset['q2'] < 6.0)]
        return dataset
    elif num == 9:
        dataset = dataset[(dataset['q2'] > 15.0) & (dataset['q2'] < 17.0)]
        return dataset


def legendre_series(ci, ctl, cj=1, ctk=1, cm=1, phi=1, cn=1, q2=1):
    """
    ci blah blah is the summation series coeff. The size of the input array dictates the order of the polynomial
    NOTE: Order starts from 0 eg ci=[5] returns y=5 as the polynomial, 3rd order needs array with 4 elements etc
    ctk, ctl... is the argument of the legendre poly
    """
    sum1 = np.polynomial.legendre.Legendre(ci)
    sum2 = np.polynomial.legendre.Legendre(cj)
    sum3 = np.polynomial.legendre.Legendre(cm)
    sum4 = np.polynomial.legendre.Legendre(cn)
    return sum1(ctl) * sum2(ctk) * sum3(phi) * sum4(q2)


def log_likelihood(c0, c1, c2, c3, c4):
    a = np.polynomial.Legendre([c0, c1, c2, c3, c4])
    return -np.sum(np.log(a(ctl)))


def fourth_poly(x, a0, a1, a2, a3, a4):
    return (a4 * x ** 4) + (a3 * x ** 3) + (a2 * x ** 2) + (a1 * x) + a0


#%%
ds = pd.read_pickle(r'year3-problem-solving\total_dataset.pkl')

# choose candidates with one muon PT > 1.7GeV
PT_mu_filter = (ds['mu_minus_PT'] >= 1.7 * (10 ** 3)) | (ds['mu_plus_PT'] >= 1.7 * (10 ** 3))

# Selected B0_IPCHI2<9  (3 sigma)
IP_B0_filter = ds['B0_IPCHI2_OWNPV'] < 9

# should be numerically similar to number of degrees of freedom for the decay (5)
end_vertex_chi2_filter = ds['B0_ENDVERTEX_CHI2'] < 6

# At least one of the daughter particles should have IPCHI2>16 (4 sigma)
daughter_IP_chi2_filter = (ds['mu_minus_PT'] >= 16) | (ds['mu_plus_PT'] >= 16)

# B0 should travel about 1cm (Less sure about this one - maybe add an upper limit?)
flight_distance_B0_filter = ds['B0_FD_OWNPV'] > 0.5 * (10 ** 1)

# cos(DIRA) should be close to 1
DIRA_angle_filter = ds['B0_DIRA_OWNPV'] > 0.99999

# Remove J/psi peaking background
Jpsi_filter = (ds["q2"] <= 8) | (ds["q2"] >= 11)

# Remove psi(2S) peaking background
psi2S_filter = (ds["q2"] <= 12.5) | (ds["q2"] >= 15)

# Possible pollution from Bo -> K*0 psi(-> mu_plus mu_minus)
phi_filter = (ds['q2'] <= 0.98) | (ds['q2'] >= 1.1)

# Pion likely to be kaon
pi_to_be_K_filter = ds["Pi_MC15TuneV1_ProbNNk"] < 0.95

# Kaon likely to be pion
K_to_be_pi_filter = ds["K_MC15TuneV1_ProbNNpi"] < 0.95

# pion likely to be proton
pi_to_be_p_filter = ds["Pi_MC15TuneV1_ProbNNp"] < 0.9

#Applying filters (you can remove any filter to play around with them)
ds_filtered = ds[
    end_vertex_chi2_filter
    & daughter_IP_chi2_filter
    & flight_distance_B0_filter
    & DIRA_angle_filter
    # & Jpsi_filter
    # & psi2S_filter
    # & phi_filter
    # & pi_to_be_K_filter
    # & K_to_be_pi_filter
    # & pi_to_be_p_filter
    ]
#%%
td = pd.read_pickle(r'year3-problem-solving\total_dataset.pkl')
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
Performs curve fit for acceptance data after Gaussian elimination for initial guesses"""
ds = pd.read_pickle(r'year3-problem-solving\acceptance_mc.pkl')
ctl = bin_num(ds, 8)['costhetal']  # Loads the costhetal for the 1 < q2 < 6 bin
heights, edges, _ = plt.hist(ctl, bins=100, density=True, histtype='step', label='Data')
x = np.array([(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)])  # Finds centre of each HISTOGRAM bin

x_array = np.array([[x[0] ** i for i in range(5)],  # Defines the matrix for costhetal
                    [(x[len(edges) // 4]) ** i for i in range(5)],
                    [(x[len(edges) // 2]) ** i for i in range(5)],
                    [(x[int(len(edges) // 1.2)]) ** i for i in range(5)],
                    [x[-1] ** i for i in range(5)]])

y_array = np.array([[heights[0]],  # Defines matrix/vector for the histogram height
                    [heights[len(edges) // 4]],
                    [heights[len(edges) // 2]],
                    [heights[int(len(edges) // 1.2)]],
                    [heights[-1]]])

coeff = np.linalg.solve(x_array, y_array)  # Solves for the coeffs of the polynomial

coeff, cov = curve_fit(fourth_poly, x, heights, p0=coeff)
plt.plot(x, fourth_poly(x, *coeff), label='Fit')
plt.plot(x, heights / fourth_poly(x, *coeff), label=r'$\frac{Data}{Fit}$')
plt.title('Acceptance_mc')
plt.xlabel(r'$cos(\theta_l)$')
plt.ylabel('Number')
plt.legend()

############################# Old unused code ########################################

# plt.plot(np.linspace(-1,1, 500), [log_likelihood(0.6, -0.37, 0.43, 0.38, i) for i in np.linspace(-1,1, 500)])
# log_likelihood.errordef = Minuit.LIKELIHOOD
# m = Minuit(log_likelihood, 0.6, -0.37, 0.43, 0.38, -0.88)
# m.migrad()
# param = m.values
# plt.plot(x, legendre_series(param, x))
