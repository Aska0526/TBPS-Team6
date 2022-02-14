#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
#Enter your data path here:
DATA_PATH = "Data/total_dataset.pkl"
# DATA_PATH = "Data/signal.pkl"
# DATA_PATH = "Data/acceptance_mc.pkl"

ds = pd.read_pickle(DATA_PATH)
#%%
# choose candidates with one muon PT > 1.7GeV
PT_mu_filter = (ds['mu_minus_PT'] >= 1.7*(10**3)) | (ds['mu_plus_PT'] >= 1.7*(10**3))

# Selected B0_IPCHI2<9  (3 sigma) 
IP_B0_filter = ds['B0_IPCHI2_OWNPV'] < 9

# should be numerically similar to number of degrees of freedom for the decay (5) 
end_vertex_chi2_filter = ds['B0_ENDVERTEX_CHI2'] < 6

# At least one of the daughter particles should have IPCHI2>16 (4 sigma) 
daughter_IP_chi2_filter = (ds['mu_minus_PT'] >= 16) | (ds['mu_plus_PT'] >= 16)

# B0 should travel about 1cm (Less sure about this one - maybe add an upper limit?) 
flight_distance_B0_filter = ds['B0_FD_OWNPV'] > 0.5*(10**1)

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

#%%
#Applying filters (you can remove any filter to play around with them)
ds_filtered = ds[
    end_vertex_chi2_filter
    & daughter_IP_chi2_filter
    & flight_distance_B0_filter 
    & DIRA_angle_filter
    & Jpsi_filter
    & psi2S_filter
    & phi_filter
    # & pi_to_be_K_filter
    # & K_to_be_pi_filter
    # & pi_to_be_p_filter
]

ds_filtered.to_pickle(f"Output_Data/dataset_manual_filter_1.pkl")

#%%
plt.hist(ds_filtered["B0_MM"], bins=100)
plt.xlabel("B0 mass / MeV")
plt.ylabel("Number of Candidates")
plt.title(f"Filtered")
# plt.savefig(f"Output/B0mm_filtered_manual_1_acceptance_mc.pdf")
# plt.savefig(f"Output/B0mm_filtered_manual_1_acceptance_mc.png", dpi=1000)
plt.show()
#%%
plt.hist(ds["B0_MM"], bins=100)
plt.xlabel("B0 mass / MeV")
plt.ylabel("Number of Candidates")
plt.title(f"Unfiltered")
# plt.savefig(f"Output/B0mm_unfiltered_acceptance_mc.pdf")
# plt.savefig(f"Output/B0mm_unfiltered_acceptance_mc.png", dpi=1000)
plt.show()
#%%
plt.hist(ds_filtered["costhetal"], bins=100)
plt.xlabel(r"$cos(\theta_l)$")
plt.ylabel("Number of Candidates")
plt.title(f"Filtered")
# plt.savefig(f"Output/costhetal_filtered_manual_1_acceptance_mc.pdf")
# plt.savefig(f"Output/costhetal_filtered_manual_1_acceptance_mc.png", dpi=1000)
plt.show()
#%%
plt.hist(ds["costhetal"], bins=100)
plt.xlabel(r"$cos(\theta_l)$")
plt.ylabel("Number of Candidates")
plt.title(f"Unfiltered")
# plt.savefig(f"Output/costhetal_unfiltered_acceptance_mc.pdf")
# plt.savefig(f"Output/costhetal_unfiltered_acceptance_mc.png", dpi=1000)
plt.show()
#%%
plt.hist(ds_filtered["costhetak"], bins=100)
plt.xlabel(r"$cos(\theta_k)$")
plt.ylabel("Number of Candidates")
plt.title(f"Filtered")
# plt.savefig(f"Output/costhetak_filtered_manual_1_acceptance_mc.pdf")
# plt.savefig(f"Output/costhetak_filtered_manual_1_acceptance_mc.png", dpi=1000)
plt.show()

#%%
plt.hist(ds["costhetak"], bins=100)
plt.xlabel(r"$cos(\theta_k)$")
plt.ylabel("Number of Candidates")
plt.title(f"Unfiltered")
# plt.savefig(f"Output/costhetak_unfiltered_acceptance_mc.pdf")
# plt.savefig(f"Output/costhetak_unfiltered_acceptance_mc.png", dpi=1000)
plt.show()
#%%