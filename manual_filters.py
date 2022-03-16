#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
#Enter your data path here:
DATA_PATH = "Data/total_dataset.pkl"
# DATA_PATH = "Data/signal.pkl"
# DATA_PATH = "Data/acceptance_mc.pkl"

df = pd.read_pickle(DATA_PATH)
#%%
# choose candidates with one muon PT > 1.7GeV
PT_mu_filter = (df['mu_minus_PT'] >= 1.5 * (10 ** 3)) | (df['mu_plus_PT'] >= 1.5 * (10 ** 3))

# DOI:10.1007/JHEP10(2018)047
PT_K_filter = (df['K_PT'] >= 0.5 * (10 ** 3))

PT_Pi_filter = (df['Pi_PT'] >= 0.5 * (10 ** 3))

# Selected B0_IPCHI2<9  (3 sigma) 
IP_B0_filter = df['B0_IPCHI2_OWNPV'] < 9

#arxiv:1112.3515v3
Kstarmass = (df['Kstar_MM'] <= 992) & (df['Kstar_MM'] >= 792)

# should be numerically similar to number of degrees of freedom for the decay (5)
# Physics Letters B 753 (2016) 424–448
# a track fit  per degree of freedom less than 1.8,
end_vertex_chi2_filter = df['B0_ENDVERTEX_CHI2'] < 10

# At least one of the daughter particles should have IPCHI2>16 (4 sigma)
daughter_IP_chi2_filter = (df['mu_minus_IPCHI2_OWNPV'] >= 16) | (df['mu_plus_IPCHI2_OWNPV'] >= 16)

# B0 should travel about 1cm (Less sure about this one - maybe add an upper limit?) 
flight_distance_B0_filter = (df['B0_FD_OWNPV'] <= 500) & (df['B0_FD_OWNPV'] >= 8)

# cos(DIRA) should be close to 1
# Physics Letters B 753 (2016) 424–448
DIRA_angle_filter = df['B0_DIRA_OWNPV'] > 0.9994

# Remove J/psi peaking background
Jpsi_filter = (df["q2"] <= 8) | (df["q2"] >= 11)

# Remove psi(2S) peaking background
psi2S_filter = (df["q2"] <= 12.5) | (df["q2"] >= 15)

# Possible pollution from Bo -> K*0 psi(-> mu_plus mu_minus)
phi_filter = (df['q2'] <= 0.98) | (df['q2'] >= 1.1)

#%%
#Applying filters (you can remove any filter to play around with them)
df_filtered = df[
    end_vertex_chi2_filter
    & daughter_IP_chi2_filter
    & flight_distance_B0_filter
    & DIRA_angle_filter
    & Kstarmass
    & Jpsi_filter
    & psi2S_filter
    & phi_filter
    & IP_B0_filter
    & PT_mu_filter
    & PT_K_filter
    & PT_Pi_filter
]

df_filtered.to_pickle(f"Output_Data/dataset_manual_filter_3.pkl")

#%%
plt.hist(df_filtered["B0_MM"], bins=100)
plt.xlabel("B0 mass / MeV")
plt.ylabel("Number of Candidates")
plt.title(f"Filtered")
# plt.savefig(f"Output/B0mm_filtered_manual_1_acceptance_mc.pdf")
# plt.savefig(f"Output/B0mm_filtered_manual_1_acceptance_mc.png", dpi=1000)
plt.show()
#%%
plt.hist(df["B0_MM"], bins=100)
plt.xlabel("B0 mass / MeV")
plt.ylabel("Number of Candidates")
plt.title(f"Unfiltered")
# plt.savefig(f"Output/B0mm_unfiltered_acceptance_mc.pdf")
# plt.savefig(f"Output/B0mm_unfiltered_acceptance_mc.png", dpi=1000)
plt.show()
#%%
plt.hist(df_filtered["costhetal"], bins=100)
plt.xlabel(r"$cos(\theta_l)$")
plt.ylabel("Number of Candidates")
plt.title(f"Filtered")
# plt.savefig(f"Output/costhetal_filtered_manual_1_acceptance_mc.pdf")
# plt.savefig(f"Output/costhetal_filtered_manual_1_acceptance_mc.png", dpi=1000)
plt.show()
#%%
plt.hist(df["costhetal"], bins=100)
plt.xlabel(r"$cos(\theta_l)$")
plt.ylabel("Number of Candidates")
plt.title(f"Unfiltered")
# plt.savefig(f"Output/costhetal_unfiltered_acceptance_mc.pdf")
# plt.savefig(f"Output/costhetal_unfiltered_acceptance_mc.png", dpi=1000)
plt.show()
#%%
plt.hist(df_filtered["costhetak"], bins=100)
plt.xlabel(r"$cos(\theta_k)$")
plt.ylabel("Number of Candidates")
plt.title(f"Filtered")
# plt.savefig(f"Output/costhetak_filtered_manual_1_acceptance_mc.pdf")
# plt.savefig(f"Output/costhetak_filtered_manual_1_acceptance_mc.png", dpi=1000)
plt.show()

#%%
plt.hist(df["costhetak"], bins=100)
plt.xlabel(r"$cos(\theta_k)$")
plt.ylabel("Number of Candidates")
plt.title(f"Unfiltered")
# plt.savefig(f"Output/costhetak_unfiltered_acceptance_mc.pdf")
# plt.savefig(f"Output/costhetak_unfiltered_acceptance_mc.png", dpi=1000)
plt.show()
#%%