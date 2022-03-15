import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from iminuit import Minuit
from scipy.integrate import trapezoid
from pathlib import Path
from mpl_toolkits import mplot3d


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
        dataset = dataset[(dataset['q2'] >= 0.1) & (dataset['q2'] <= 0.98)]
        return dataset
    elif num == 1:
        dataset = dataset[(dataset['q2'] >= 1.1) & (dataset['q2'] <= 2.5)]
        return dataset
    elif num == 2:
        dataset = dataset[(dataset['q2'] >= 2.5) & (dataset['q2'] <= 4.0)]
        return dataset
    elif num == 3:
        dataset = dataset[(dataset['q2'] >= 4.0) & (dataset['q2'] <= 6.0)]
        return dataset
    elif num == 4:
        dataset = dataset[(dataset['q2'] >= 6.0) & (dataset['q2'] <= 8.0)]
        return dataset
    elif num == 5:
        dataset = dataset[(dataset['q2'] >= 15.0) & (dataset['q2'] <= 17.0)]
        return dataset
    elif num == 6:
        dataset = dataset[(dataset['q2'] >= 17.0) & (dataset['q2'] <= 19.0)]
        return dataset
    elif num == 7:
        dataset = dataset[(dataset['q2'] >= 11.0) & (dataset['q2'] <= 12.5)]
        return dataset
    elif num == 8:
        dataset = dataset[(dataset['q2'] >= 1.0) & (dataset['q2'] <= 6.0)]
        return dataset
    elif num == 9:
        dataset = dataset[(dataset['q2'] >= 15.0) & (dataset['q2'] <= 17.9)]
        return dataset


def poly(x, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12):  # doesnt get much better after order 6
    """
    set the jth coeff of the power series by aj
    has to do it this way since curve_fit unpacks p0 automatically
    np.polynomial.Polynomial sets up the power series in ascending order
    """
    return np.polynomial.Polynomial([a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12])(x)


def d2gamma_p_d2q2_dcostheta(fl, afb, cos_theta_l, bn):
    ctl = np.array(cos_theta_l)
    c2tl = 2 * (ctl ** 2) - 1
    array = (3 / 8) * ((3 / 2) - (fl * 1 / 2) + (c2tl * 1 / 2) * (1 - 3 * fl) + (8 / 3) * afb * ctl)
    # if np.min(array) <= 0.0:
        # del_index = np.where(array <= 0.0)  # Method 1 (delete invalid ones)
        # array = np.delete(array, del_index)
        # ctl = np.delete(ctl, del_index)
        # array = array - (np.min(array))  # Method 2 (shift up)
        # array = abs(array)  # Method 3 (take absolute)
    A0 = trapezoid(array, ctl)
    array *= poly(ctl, *cof_mt[bn])
    A1 = trapezoid(array, ctl)
    normalised_array = A0 * array / A1
    return normalised_array


def log_likelihood(fl, afb, _bin):
    ctl = bins[int(_bin)]
    normalised_scalar_array = d2gamma_p_d2q2_dcostheta(fl=fl, afb=afb, cos_theta_l=ctl, bn=int(_bin))
    return - np.sum(np.log(normalised_scalar_array))


afb_val_theo = [-0.097, -0.138, -0.017, 0.122, 0.240, 0.402, 0.318, 0.391, 0.005, 0.368]
afb_err_theo = [0.008, 0.032, 0.029, 0.040, 0.047, 0.030, 0.034, 0.024, 0.028, 0.032]

fl_val_theo = [0.296, 0.76, 0.796, 0.711, 0.607, 0.348, 0.328, 0.435, 0.748, 0.34]
fl_err_theo = [0.05, 0.04, 0.03, 0.05, 0.05, 0.04, 0.03, 0.04, 0.04, 0.02]
#%%
# Reads the total dataset and apply some manual cuts
ds = pd.read_pickle(Path(r'year3-problem-solving/XGB_filter_signal_td_depth_15_equal_oversampling.pkl'))
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
    Jpsi_filter
    & psi2S_filter
    # & end_vertex_chi2_filter
    # & daughter_IP_chi2_filter
    # & flight_distance_B0_filter
    # & DIRA_angle_filter

    # & phi_filter
    #& pi_to_be_K_filter
    #& K_to_be_pi_filter
    #& pi_to_be_p_filter
    ]

#%%
# Fits and plots B0_MM distribution

B_mass = ds_filtered['B0_MM']
height, edges, _ = plt.hist(B_mass, bins=200, range=[5180, 5700], label='data')
plt.xlabel(r'B0 mass $\left(\frac{MeV}{c^2}\right)$')
plt.ylabel('Number')

p0 = [2100, 0.0003, 5180, 50, 2000, 5300]
x = np.array([(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)])
popt, cov = curve_fit(combined, x, height, p0=p0)
plt.plot(x, combined(x, *popt), label='Fit')
plt.legend()
plt.show()

#%%
# Fits the acceptance dataset with 6th order poly
amc = pd.read_pickle(Path(r'year3-problem-solving/XGB_filter_signal_acceptancemc_depth_15_equal_oversampling.pkl'))

cof_mt = []

# Do this for all 10 predefined bins
order = 12
for bn in range(10):
    plt.figure()
    ctl = bin_num(amc, bn)['costhetal']
    heights, edges, _ = plt.hist(ctl, bins=70, density=True, histtype='step', label='Data')
    plt.close()  # Comment out this if u want to see the histograms
    x = np.array([(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)])  # Finds centre of each HISTOGRAM bin

    x_array = np.array([[x[0] ** i for i in range(order + 1)],  # Defines the matrix for costhetal.
                        [(x[int(len(edges) // 10)]) ** i for i in range(order + 1)],
                        [(x[int(len(edges) // 2)]) ** i for i in range(order + 1)],
                        [(x[int(len(edges) // -5)]) ** i for i in range(order + 1)],
                        [(x[int(len(edges) // -7)]) ** i for i in range(order + 1)],
                        [(x[int(len(edges) // -1.1)]) ** i for i in range(order + 1)],
                        [(x[int(len(edges) // 4)]) ** i for i in range(order + 1)],
                        [(x[int(len(edges) // 6)]) ** i for i in range(order + 1)],
                        [(x[int(len(edges) // 9)]) ** i for i in range(order + 1)],
                        [(x[int(len(edges) // -3)]) ** i for i in range(order + 1)],
                        [(x[int(len(edges) // -9)]) ** i for i in range(order + 1)],
                        [(x[int(len(edges) // -3.5)]) ** i for i in range(order + 1)],
                        [x[-1] ** i for i in range(order + 1)]])

    y_array = np.array([
        [heights[0]],  # Defines matrix/vector for the histogram height. NOTE should match the order of poly
        [heights[int(len(edges) // 10)]],
        [heights[int(len(edges) // 2)]],
        [heights[int(len(edges) // -5)]],
        [heights[int(len(edges) // -7)]],
        [heights[int(len(edges) // -1.1)]],
        [heights[int(len(edges) // 4)]],  # Comment out for appropriate order
        [heights[int(len(edges) // 6)]],
        [heights[int(len(edges) // 9)]],
        [heights[int(len(edges) // -3)]],
        [heights[int(len(edges) // -9)]],
        [heights[int(len(edges) // -3.5)]],
        [heights[-1]]])

    coeffs = np.linalg.solve(x_array, y_array)[:, 0]  # Solves for the coeffs of the polynomial
    coeffs, cov = curve_fit(poly, x, heights, p0=coeffs)
    norm = trapezoid(poly(x, *coeffs), x)
    cof_mt.append(coeffs / norm)

    # Uncomment if u want to inspect the fit
    # plt.plot(x, poly(x, *coeffs), label='Fit')
    # plt.plot(x, heights / poly(x, *coeffs), label=r'$\frac{Data}{Fit}$')
    # plt.title(f'Acceptance bin{bn}')
    # plt.xlabel(r'$cos(\theta_l)$')
    # plt.ylabel('Number')
    # plt.legend()
    # plt.show()

#%%
# Performs iMinuit fit for Afb, FL

bins = []
for i in range(10):
    bins.append(sorted(bin_num(ds_filtered, i)['costhetal'].transpose().to_numpy()))

bin_number_to_check = 4  # bin that we want to check in more details
bin_results_to_check = None

log_likelihood.errordef = Minuit.LIKELIHOOD
decimal_places = 3
starting_point = [0.8, 0.1]
chi_fl, chi_afb = [], []

xx, yy = np.meshgrid(np.linspace(-1.0, 1.0, 10), np.linspace(0.0, 1.0, 10))

# for a in xx[0, :]:
#     for f in yy[:, 0]:
fls, fl_errs = [], []
afbs, afb_errs = [], []
dummy_fl, dummy_afb = [], []
for i in range(10):
    m = Minuit(log_likelihood, fl=starting_point[0], afb=starting_point[1], _bin=i)
    m.fixed['_bin'] = True
    m.limits = ((0.0, 1.0), (-1.0, 1.0), None)
    m.migrad()
    m.hesse()
    if i == bin_number_to_check:
        bin_results_to_check = m
    fls.append(m.values[0])
    afbs.append(m.values[1])
    # dummy_fl.append((fls[i] - fl_val_theo[i]) ** 2)
    # dummy_afb.append((afbs[i] - afb_val_theo[i]) ** 2)
    fl_errs.append(m.errors[0])
    afb_errs.append(m.errors[1])
    print(f"Bin {i}: {np.round(fls[i], decimal_places)} +/- {np.round(fl_errs[i], decimal_places)},",
          f"{np.round(afbs[i], decimal_places)} +/- {np.round(afb_errs[i], decimal_places)}. valid?: {m.fmin.is_valid}")
# chi_fl.append(np.sum(dummy_fl))
# chi_afb.append(np.sum(dummy_afb))
#%% check afb and fl
# =============================================================================
# plt.figure(figsize=(8, 5))
# plt.subplot(221)
# bin_results_to_check.draw_mnprofile('afb', bound=3)
# plt.subplot(222)
# bin_results_to_check.draw_mnprofile('fl', bound=3)
# plt.tight_layout()
# plt.show()
#
# =============================================================================

#%% check pdf
# =============================================================================
# bin_to_plot = bin_number_to_check
# number_of_bins_in_hist = 70
# cos_theta_l_bin = bins[bin_to_plot]
# plt.figure()
# hist, _bins, _ = plt.hist(cos_theta_l_bin, bins=number_of_bins_in_hist)
# x = np.linspace(-1, 1, number_of_bins_in_hist)
# pdf_multiplier = np.sum(hist) * (np.max(cos_theta_l_bin) - np.min(cos_theta_l_bin)) / number_of_bins_in_hist
# y = d2gamma_p_d2q2_dcostheta(fl=fls[bin_to_plot], afb=afbs[bin_to_plot], cos_theta_l=x, bn=bin_to_plot) * pdf_multiplier
# plt.plot(x, y, label=f'Fit for bin {bin_to_plot}')
# plt.xlabel(r'$cos(\theta_l)$')
# plt.ylabel(r'Number of candidates')
# plt.legend()
# plt.grid()
# plt.show()
# =============================================================================

# Plots fitted values
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
# ax1.errorbar(np.linspace(0, len(bins) - 1, len(bins)), fls, yerr=fl_errs, fmt='o', markersize=2, label=r'$F_L$',
#              color='red')
# ax2.errorbar(np.linspace(0, len(bins) - 1, len(bins)), afbs, yerr=afb_errs, fmt='o', markersize=2, label=r'$A_{FB}$',
#              color='red')
# ax1.grid()
# ax2.grid()
# ax1.set_ylabel(r'$F_L$')
# ax2.set_ylabel(r'afb')
# ax1.set_xlabel(r'Bin number')
# ax2.set_xlabel(r'Bin number')
# plt.tight_layout()
# plt.show()


# Plots theoretical values
plt.subplots(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.errorbar(np.linspace(0, len(bins) - 1, len(bins)), fls, yerr=fl_errs, fmt='o', markersize=5, label=r'fit',
             color='red')
plt.errorbar(np.linspace(0, len(bins) - 1, len(bins)), fl_val_theo, yerr=fl_err_theo, fmt='x', markersize=6,
             label=r'prediction', color='blue')
plt.grid()
plt.legend()
plt.ylabel(r'$F_L$')
plt.xlabel(r'Bin number')

plt.subplot(1, 2, 2)
plt.errorbar(np.linspace(0, len(bins) - 1, len(bins)), afbs, yerr=afb_errs, fmt='o', markersize=5, label=r'fit',
             color='red')
plt.errorbar(np.linspace(0, len(bins) - 1, len(bins)), afb_val_theo, yerr=afb_err_theo, fmt='x', markersize=6,
             label=r'prediction', color='blue')
plt.grid()
plt.legend()
plt.ylabel(r'afb')
plt.xlabel(r'Bin number')
plt.suptitle('Total dataset (filtered)', fontsize=20)
plt.show()

#%%
ax = plt.axes(projection='3d')
ax.scatter3D(xx.reshape(-1), yy.reshape(-1), np.add(chi_fl, chi_afb), 'r', s=10, c='r')
ax.set_xlabel('afb')
ax.set_ylabel('fl')
ax.set_zlabel('sum of mean squared')
ax.plot(xx.reshape(-1), yy.reshape(-1), 'kx', zdir='z', markersize=1)

#%%
'''
NOT COMPLETE    
'''
fl = np.linspace(-1.0, 1.0, 5)
afb = np.linspace(-1.0, 1.0, 5)
xx, yy = np.meshgrid(afb, fl)
xx = xx.reshape(-1)
yy = yy.reshape(-1)
z = []
for x in xx:
    for y in yy:
        z.append(log_likelihood(x, y, 0))

ax = plt.axes(projection='3d')
ax.plot_wireframe(xx, yy, z)
