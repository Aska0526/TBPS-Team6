#%%
import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt
from scipy.integrate import quad
#%%
F_l_fit = np.array([0.491, 0.827, 0.678, 0.48, 0.416, 0.253, 0.187, 0.268, 0.527, 0.251])
F_l_fit_err = np.array([0.168, 0.206, 0.167, 0.133, 0.124, 0.105, 0.149, 0.102, 0.097, 0.091])

A_fb_fit = np.array([-0.046, -0.182, -0.074, 0.124, 0.104, 0.425, 0.414, 0.378, -0.006, 0.426])
A_fb_fit_err = np.array([0.085, 0.075, 0.095, 0.078, 0.068, 0.059, 0.088, 0.064, 0.055, 0.053])

F_l_theory = np.array([0.296, 0.76, 0.796, 0.711, 0.607, 0.348, 0.328, 0.435, 0.748, 0.34])
F_l_theory_err = np.array([0.05, 0.04, 0.03, 0.05, 0.05, 0.04, 0.03, 0.04, 0.04, 0.02])

A_fb_theory = np.array([-0.097, -0.138, -0.017, 0.122, 0.240, 0.402, 0.318, 0.391, 0.005, 0.368])
A_fb_theory_err = np.array([0.008, 0.032, 0.029, 0.040, 0.047, 0.030, 0.034, 0.024, 0.028, 0.032])
#%%
n_sigma_F_l = (F_l_theory - F_l_fit) / (F_l_fit_err)

n_sigma_A_fb = (A_fb_theory - A_fb_fit) / (A_fb_fit_err)

np.savetxt("Output_Data/n_sigma_F_l.txt", n_sigma_F_l, fmt="%.18f")
np.savetxt("Output_Data/n_sigma_A_fb.txt", n_sigma_A_fb, fmt="%.18f")


#%%
x = np.arange(-1000, 1000, 0.001)

def gaussian(x):
    return scipy.stats.norm.pdf(x, 0, 1)
#%%
ps_F_l = np.array([quad(gaussian, -np.inf, n_sigma)[0] for n_sigma in n_sigma_F_l])
ps_A_fb = np.array([quad(gaussian, -np.inf, n_sigma)[0] for n_sigma in n_sigma_A_fb])

np.savetxt("Output_Data/ps_F_l.txt", ps_F_l, fmt="%.18f")
np.savetxt("Output_Data/ps_A_fb.txt", ps_A_fb, fmt="%.18f")
#%%
p_range_F_l = [np.amin(ps_F_l), np.amax(ps_F_l)]
p_range_A_fb = [np.amin(ps_A_fb), np.amax(ps_A_fb)]

np.savetxt("Output_Data/p_range_F_l.txt", p_range_F_l, fmt="%.18f")
np.savetxt("Output_Data/p_range_A_fb.txt", p_range_A_fb, fmt="%.18f")
#%%
def chi_squared(theory, fit, fit_err):
    p = (fit - theory) / fit_err
    return sum(p ** 2)

#%%
chi_squared_F_l = chi_squared(F_l_theory, F_l_fit, F_l_fit_err)
#%%