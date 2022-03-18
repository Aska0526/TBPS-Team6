#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import xgboost as xgb
#%%
DATA_PATH = "Data/total_dataset.pkl"
df = pd.read_pickle(DATA_PATH)
df["B0_PT"] = df["mu_minus_PT"] + df["mu_plus_PT"] + df["K_PT"] + df["Pi_PT"]
df["B0_PX"] = df["mu_minus_PX"] + df["mu_plus_PX"] + df["K_PX"] + df["Pi_PX"]
df["B0_PY"] = df["mu_minus_PY"] + df["mu_plus_PY"] + df["K_PY"] + df["Pi_PY"]
df["B0_PZ"] = df["mu_minus_PZ"] + df["mu_plus_PZ"] + df["K_PZ"] + df["Pi_PZ"]
df["B0_PE"] = df["mu_minus_PE"] + df["mu_plus_PE"] + df["K_PE"] + df["Pi_PE"]
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
df_filtered_manual = df[
    Jpsi_filter
    & psi2S_filter
    & phi_filter

    & end_vertex_chi2_filter
    & daughter_IP_chi2_filter
    & flight_distance_B0_filter
    & DIRA_angle_filter
    & Kstarmass

    & IP_B0_filter
    & PT_mu_filter
    & PT_K_filter
    & PT_Pi_filter
]
#%%
# MODEL_PATH = "Model/signal_td_balanced_oversampling_noq2_treedepth_12_eta_0.1_state_6.model"
# MODEL_PATH = "Model/signal_td_balanced_oversampling_treedepth_12_eta_0.1_state_6.model"
MODEL_PATH = "Model/signal_td_reconstruction_balanced_oversampling_treedepth_12_eta_0.1_state_6.model"
bst = xgb.XGBClassifier()
bst.load_model(MODEL_PATH)
#%%
if "noq2" in MODEL_PATH:
    drop_columns = ["year", "B0_MM", "q2"]
else:
    drop_columns = ["year", "B0_MM"]

df_test_xgb = df_filtered_manual.drop(columns=drop_columns)

df_clf_filtered_xgb = df_test_xgb.loc[bst.predict(df_test_xgb), :]
df_clf_filtered = df_filtered_manual.loc[df_clf_filtered_xgb.index, :]

print(len(df_clf_filtered) / len(df_filtered_manual))
print(len(df_filtered_manual) / len(df))
print(len(df_clf_filtered) / len(df))

# df_clf_filtered.to_pickle(f"Output_Data/XGB_filter_signal_td_totaldataset_noq2_depth_9_equal_oversampling.pkl")
# df_clf_filtered.to_pickle(f"Output_Data/XGB_filter_signal_td_totaldataset_3peakcuts_depth_12_equal_oversampling.pkl")
# df_clf_filtered.to_pickle(f"Output_Data/XGB_filter_signal_td_reconstruction_acceptancemc_noflightdistanceB0_depth_12_equal_oversampling.pkl")


#%%
plt.hist(df_clf_filtered["B0_MM"], bins=100, label="XG Boosted Decision Tree", density=True)
# plt.hist(df["B0_MM"], bins=100, label="No Cuts", density=True, alpha=0.5)
plt.hist(df_filtered_manual["B0_MM"], bins=100, label="Manual Cuts only", density=True, alpha=0.5)
plt.xlabel("B0 mass / MeV")
plt.ylabel("Number of Candidates")
plt.title("Applying BDT to acceptance_mc")
plt.legend()
plt.show()
#%%
df_filtered_1 = df[
    Jpsi_filter
    & psi2S_filter
    & phi_filter

    & end_vertex_chi2_filter
    & daughter_IP_chi2_filter
    & flight_distance_B0_filter
    & DIRA_angle_filter
    & Kstarmass

    & IP_B0_filter
    & PT_mu_filter
    & PT_K_filter
    & PT_Pi_filter
]

df_filtered_2 = df[
    Jpsi_filter
    & psi2S_filter
    & phi_filter

    # & end_vertex_chi2_filter
    # & daughter_IP_chi2_filter
    # & flight_distance_B0_filter
    # & DIRA_angle_filter
    # & Kstarmass

    # & IP_B0_filter
    # & PT_mu_filter
    # & PT_K_filter
    # & PT_Pi_filter
]

plt.hist(df_filtered_2["q2"], bins=100, label="all cuts", density=True, alpha=0.5)
plt.hist(df_filtered_1["q2"], bins=100, label="3 basic manual cuts", density=True, alpha=0.5)
plt.hist(df_clf_filtered["q2"], bins=100, label="with ML", density=True, alpha=0.5)
plt.xlabel("q2 / MeV")
plt.ylabel("Number of Candidates")
plt.xticks(np.arange(0,26))
# plt.title("Applying BDT to acceptance_mc")
plt.legend()
# plt.savefig("Output/cuts_comparison_1.jpeg", dpi=1000)
plt.show()
#%%
TEST_DATA_PATH = "Data/total_dataset.pkl"
# df_test = pd.read_pickle(TEST_DATA_PATH)
df_test = df_filtered_manual
df_cut = df_clf_filtered

bins_b = [0.1, 1.1, 2.5, 4, 6, 15, 17, 11, 1, 15]

bins_t = [0.98, 2.5, 4, 6, 8, 17, 19, 12.5, 6, 17.9]

q = [[],[],[],[],[],[], [],[],[],[]]

d = [[],[],[],[],[],[], [],[],[],[]]

df = [[],[],[],[],[],[], [],[],[],[]]

ds = df_test

# ds_filtered = df_clf_filtered
ds_filtered = df_cut

for i in range(len(bins_b)):

    q[i] = (ds["q2"] >= bins_b[i]) & (ds["q2"] < bins_t[i])

    d[i] = ds[q[i]]

    df[i] = ds_filtered[q[i]]

fig, axs = plt.subplots(nrows = 2, ncols = 5, figsize = (24,12))

titles = [
    r"$0.1 \leq q^2 < 0.98$ MeV",
    r"$1.1 \leq q^2 < 2.5$ MeV",
    r"$2.5 \leq q^2 < 4$ MeV",
    r"$4 \leq q^2 < 6$ MeV",
    r"$6 \leq q^2 < 8$ MeV",
    r"$15 \leq q^2 < 17$ MeV",
    r"$17 \leq q^2 < 19$ MeV",
    r"$11 \leq q^2 < 12.5$ MeV",
    r"$1 \leq q^2 < 6$ MeV",
    r"$15 \leq q^2 < 17.9$ MeV",
    ]

for i, ax in enumerate(axs.flat):

    ax.hist(d[i]["B0_MM"], bins=20)

    ax.hist(df[i]["B0_MM"],bins = 20)

    ax.title.set_text(titles[i])

    ax.set_ylabel("Number of Candidates")

    ax.set_xlabel("B0 mass / MeV")

    #ax.set_ylim(0, 150)
plt.savefig("Output/q2bins_XGB_signal_td_reconstruction_max_depth_12_oversampling.jpeg", dpi=1000)
plt.show()
#%%
# plt.hist(df["q2"], bins=100, label="Manual Cuts only")
plt.hist(df_clf_filtered["q2"], bins=100, label="XG Boosted Decision Tree")
plt.xlabel("q2 / MeV")
plt.ylabel("Number of Candidates")
plt.title("Applying BDT to acceptance_mc")
plt.legend()
plt.show()
#%%