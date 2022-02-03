#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from matplotlib import colors
#%%
DATA_PATH = "Data/total_dataset.pkl"
ds = pd.read_pickle(DATA_PATH)

DATA_PATH_JPSI = "Data/jpsi.pkl"
ds_Jpsi = pd.read_pickle(DATA_PATH_JPSI)
print(ds_Jpsi.columns)
#%%
plt.hist(ds["B0_MM"], bins=100)
plt.xlabel("B0 mass / MeV")
plt.ylabel("Number of Candidates")
plt.title(r"Distribution for $J_{\psi}$")
plt.show()
#%%
plt.hist(ds["q2"], bins=50)
plt.xlabel(r"$q^2$")
plt.ylabel("Number of Candidates")
plt.title(r"Unfiltered $q^2$ Distribution for total dataset")
plt.savefig("Output/q2_hist_unfiltered.pdf")
plt.show()
#%%
# ds_q2_hist = sp.stats.rv_histogram(np.histogram(ds["q2"], bins=50))


plt.hist(ds["q2"], density=True, bins=50, label="Unfiltered")
plt.hist(ds_Jpsi["q2"], density=True, bins=100, label="Filtered")
plt.xlabel(r"$q^2$ / MeV")
plt.ylabel("Probability Density")
plt.title(r"Simulated Data, $q^2$ Distribution for $J_{\psi}$")
plt.legend()
plt.show()
#%%
plt.hist(ds["J_psi_MM"], bins=20)
plt.xlabel(r"$J_\psi$ mass / MeV")
plt.ylabel("Number of Candidates")
plt.title(r"Distribution for $J_{\psi}$")
plt.show()
#%%
Jpsi_MM_minimum = 3036
Jpsi_MM_maximum = 3156

ds_filtered_Jpsi_MM = ds[(ds["J_psi_MM"] <= Jpsi_MM_minimum) | (ds["J_psi_MM"] >= Jpsi_MM_maximum)]
#%%
plt.hist(ds["B0_MM"], bins=100)
plt.hist(ds_filtered_Jpsi_MM["B0_MM"], bins=100)
plt.xlabel("B0 mass / MeV")
plt.ylabel("Number of Candidates")
plt.title(r"Filtered Distribution using $J_{\psi}$ mass")
# plt.savefig("Output/B0_MM_hist_filtered_Jpsi_MM.pdf")
# plt.savefig("Output/B0_MM_hist_filtered_Jpsi_MM.png", dpi=1000)
plt.show()

#%%
ds_Jpsi_filtered_Jpsi_MM = ds_Jpsi[(ds_Jpsi["J_psi_MM"] <= Jpsi_MM_minimum) | (ds_Jpsi["J_psi_MM"] >= Jpsi_MM_maximum)]

plt.hist(ds_Jpsi_filtered_Jpsi_MM["B0_MM"], bins=100)
plt.xlabel("B0 mass / MeV")
plt.ylabel("Number of Candidates")
# plt.title(r"Filtered Distribution for $J_{\psi}$")
plt.show()
#%%
plt.scatter(ds["B0_MM"], ds["q2"], alpha=0.01, s=5)
plt.ylabel(r"$q^2 \, / \, GeV^2$")
plt.xlabel(r"B0 mass / MeV")
plt.title("Unfiltered")
plt.savefig("Output/B0_MM_q2_unfiltered.png", dpi=1000)
plt.savefig("Output/B0_MM_q2_unfiltered.pdf")
plt.show()
#%%
Jpsi_q2_minimum = 8
Jpsi_q2_maximum = 11

Jpsi_filter = (ds["q2"] <= Jpsi_q2_minimum) | (ds["q2"] >= Jpsi_q2_maximum)

ds_filtered_Jpsi = ds[Jpsi_filter]

plt.scatter(ds_filtered_Jpsi["B0_MM"], ds_filtered_Jpsi["q2"], alpha=0.01, s=5)
plt.ylabel(r"$q^2 \, / \, GeV^2$")
plt.xlabel(r"B0 mass / MeV")
plt.title(r"$J/\psi$ Filtering, $8 < q^2 < 11 \, GeV^2$")
plt.savefig("Output/B0_MM_q2_filtered_Jpsi_8_11.png", dpi=1000)
plt.savefig("Output/B0_MM_q2_filtered_Jpsi_8_11.pdf")
plt.show()

# plt.hist(ds["B0_MM"], bins=100, label="Unfiltered")
plt.hist(ds_filtered_Jpsi["B0_MM"], bins=100, label="Filtered")
plt.xlabel("B0 mass / MeV")
plt.ylabel("Number of Candidates")
plt.title(r"$J/\psi$ Filtering, $8 < q^2 < 11 \, GeV^2$")
plt.savefig("Output/B0mm_hist_filtered_Jpsi_8_11.pdf")
plt.savefig("Output/B0mm_hist_filtered_Jpsi_8_11.png", dpi=1000)
# plt.legend()
plt.show()
#%%
psi2S_q2_minimum = 12.5
psi2S_q2_maximum = 15

psi2S_filter = (ds["q2"] <= psi2S_q2_minimum) | (ds["q2"] >= psi2S_q2_maximum)

ds_filtered_psi2S = ds[psi2S_filter]

plt.scatter(ds_filtered_psi2S["B0_MM"], ds_filtered_psi2S["q2"], alpha=0.01, s=5)
plt.ylabel(r"$q^2 \, / \, GeV^2$")
plt.xlabel(r"B0 mass / MeV")
plt.title(r"$\psi(2S)$ Filtering, $12.5 < q^2 < 15 \, GeV^2$")
plt.savefig("Output/B0_MM_q2_filtered_psi2S_12.5_15.png", dpi=1000)
plt.savefig("Output/B0_MM_q2_filtered_psi2S_12.5_15.pdf")
plt.show()

# plt.hist(ds["B0_MM"], bins=100, label="Unfiltered")
plt.hist(ds_filtered_psi2S["B0_MM"], bins=100, label="Filtered")
plt.xlabel("B0 mass / MeV")
plt.ylabel("Number of Candidates")
plt.title(r"$\psi(2S)$ Filtering, $12.5 < q^2 < 15 \, GeV^2$")
plt.savefig("Output/B0mm_hist_filtered_psi2S_12.5_15.pdf")
plt.savefig("Output/B0mm_hist_filtered_psi2S_12.5_15.png", dpi=1000)
# plt.legend()
plt.show()
#%%
ds_filtered_Jpsi_psi2S = ds[Jpsi_filter & psi2S_filter]

plt.scatter(ds_filtered_Jpsi_psi2S["B0_MM"], ds_filtered_Jpsi_psi2S["q2"], alpha=0.01, s=10)
plt.ylabel(r"$q^2 \, / \, GeV^2$")
plt.xlabel(r"B0 mass / MeV")
plt.title(r"$J/\psi$ ($8 < q^2 < 11 \, GeV^2$) and $\psi(2S)$ ($12.5 < q^2 < 15 \, GeV^2$) Filtering")
plt.savefig("Output/B0_MM_q2_filtered_Jpsi_8_11_psi2S_12.5_15.png", dpi=1000)
plt.savefig("Output/B0_MM_q2_filtered_Jpsi_8_11_psi2S_12.5_15.pdf")
plt.show()

plt.hist(ds_filtered_Jpsi_psi2S["B0_MM"], bins=100)
plt.xlabel("B0 mass / MeV")
plt.ylabel("Number of Candidates")
plt.title(r"$J/\psi$ ($8 < q^2 < 11 \, GeV^2$) and $\psi(2S)$ ($12.5 < q^2 < 15 \, GeV^2$) Filtering")
plt.savefig("Output/B0mm_hist_filtered_Jpsi_8_11_psi2S_12.5_15.pdf")
plt.savefig("Output/B0mm_hist_filtered_Jpsi_8_11_psi2S_12.5_15.png", dpi=1000)
plt.show()
#%%