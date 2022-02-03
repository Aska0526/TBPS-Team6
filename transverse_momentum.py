#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
DATA_PATH = "Data/total_dataset.pkl"
ds = pd.read_pickle(DATA_PATH)
#%%
B0_PT = ds["mu_minus_PT"] + ds["mu_plus_PT"] + ds["K_PT"] + ds["Pi_PT"]
#%%
plt.hist(B0_PT, bins=500)
plt.xlabel("Transverse Momentum / MeV")
plt.ylabel("Number of Candidates")
plt.title("Unfiltered")
plt.savefig("Output/transverse_momentum_hist_unfiltered.pdf")
plt.savefig("Output/transverse_momentum_hist_unfiltered.png", dpi=1000)
plt.show()

plt.hist(ds["B0_MM"], bins=100)
plt.xlabel("B0 mass / MeV")
plt.ylabel("Number of Candidates")
plt.title(f"Unfiltered")
plt.savefig(f"Output/B0mm_hist_pt_min_unfiltered.pdf")
plt.savefig(f"Output/B0mm_hist_pt_min_unfiltered.png", dpi=1000)
plt.show()
#%%
###
#change mass threshold here
B0_PT_minimum = 60000

#change filtered bin size here
B0_PT_filtered = B0_PT[B0_PT >= B0_PT_minimum]
###

plt.hist(B0_PT_filtered, bins=100)
plt.xlabel("Transverse Momentum / MeV")
plt.ylabel("Number of Candidates")
plt.title(f"Filtered, Minimum Transverse Momentum = {B0_PT_minimum} MeV")
# plt.savefig(f"Output/transverse_momentum_hist_min{B0_PT_minimum}.pdf")
# plt.savefig(f"Output/transverse_momentum_hist_min{B0_PT_minimum}.png", dpi=1000)
plt.show()


plt.hist(ds["B0_MM"][B0_PT_filtered.index], bins=100)
plt.xlabel("B0 mass / MeV")
plt.ylabel("Number of Candidates")
plt.title(f"Filtered, Minimum Transverse Momentum = {B0_PT_minimum} MeV")
# plt.savefig(f"Output/B0mm_hist_pt_min{B0_PT_minimum}.pdf")
# plt.savefig(f"Output/B0mm_hist_pt_min{B0_PT_minimum}.png", dpi=1000)
plt.show()

#%%
ds_filtered = ds.iloc[B0_PT_filtered.index, :]
ds_filtered.to_pickle(f"Output_Data/filtered_dataset_transverse_momentum_{B0_PT_minimum}.pkl")
#%%