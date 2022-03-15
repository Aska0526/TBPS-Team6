#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import xgboost as xgb
#%%
DATA_PATH = "Data/acceptance_mc.pkl"
df = pd.read_pickle(DATA_PATH)
df["B0_PT"] = df["mu_minus_PT"] + df["mu_plus_PT"] + df["K_PT"] + df["Pi_PT"]
df["B0_PX"] = df["mu_minus_PX"] + df["mu_plus_PX"] + df["K_PX"] + df["Pi_PX"]
df["B0_PY"] = df["mu_minus_PY"] + df["mu_plus_PY"] + df["K_PY"] + df["Pi_PY"]
df["B0_PZ"] = df["mu_minus_PZ"] + df["mu_plus_PZ"] + df["K_PZ"] + df["Pi_PZ"]
df["B0_PE"] = df["mu_minus_PE"] + df["mu_plus_PE"] + df["K_PE"] + df["Pi_PE"]
#%%
MODEL_PATH = "Model/signal_td_balanced_oversampling_treedepth_12_eta_0.1_state_6.model"
bst = xgb.XGBClassifier()
bst.load_model(MODEL_PATH)
#%%
if "noq2" in MODEL_PATH:
    drop_columns = ["year", "B0_MM", "q2"]
else:
    drop_columns = ["year", "B0_MM"]

df_test_xgb = df.drop(columns=drop_columns)

df_clf_filtered_xgb = df_test_xgb.loc[bst.predict(df_test_xgb), :]
df_clf_filtered = df.loc[df_clf_filtered_xgb.index, :]

# df_clf_filtered.to_pickle(f"Output_Data/XGB_filter_signal_acceptancemc_depth_12_equal_oversampling.pkl")
#%%
plt.hist(df["B0_MM"], bins=100, label="Manual Cuts only")
plt.hist(df_clf_filtered["B0_MM"], bins=100, label="XG Boosted Decision Tree")
plt.xlabel("B0 mass / MeV")
plt.ylabel("Number of Candidates")
plt.title("Applying BDT to acceptance_mc")
plt.legend()
plt.show()
#%%