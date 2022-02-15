#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#%%
TEST_DATA_PATH = "Output_Data/dataset_manual_filter_1.pkl"
df_test = pd.read_pickle(TEST_DATA_PATH)
df_test_xgb = df_test.drop(columns=["year"])

TOTAL_DATA_PATH = "Data/total_dataset.pkl"
df_total = pd.read_pickle(TOTAL_DATA_PATH)

# SIGNAL_DATA_PATH = "Data/signal.pkl"
SIGNAL_DATA_PATH = "Data/jpsi.pkl"
df_signal = pd.read_pickle(SIGNAL_DATA_PATH)
#%%
mass_filter = df_total['B0_MM'] >= 5350

df_background = df_total[mass_filter]
#%%
df_signal["signal"] = np.ones(len(df_signal), dtype=bool)
df_background["signal"] = np.zeros(len(df_background), dtype=bool)

df_train = pd.concat([df_signal, df_background], ignore_index=True)
df_train = df_train.drop(columns=["year"])
#%%
x, y = df_train[df_train.columns[~df_train.columns.isin(["signal"])]].values, df_train["signal"].values

dmatrix_train = xgb.DMatrix(data=x, label=y)
#%%

x_train, x_validate, y_train, y_validate = train_test_split(
    df_train[df_train.columns[~df_train.columns.isin(["signal"])]], 
    df_train["signal"],
    test_size=0.2,
    random_state=6)

#%%
xg_clf = xgb.XGBClassifier(verbosity=2)
xg_clf.fit(x_train, y_train)
#%%
y_pred = xg_clf.predict(x_validate)
accuracy_score(y_validate, y_pred)
#%%
df_clf_filtered_xgb = df_test_xgb.loc[xg_clf.predict(df_test_xgb), :]
df_clf_filtered = df_test.loc[df_clf_filtered_xgb.index, :]
#%%
plt.hist(df_test["B0_MM"], bins=100, label="Manual Cuts only")
plt.hist(df_clf_filtered["B0_MM"], bins=100, label="XG Boosted Decision Tree")
plt.xlabel("B0 mass / MeV")
plt.ylabel("Number of Candidates")
plt.legend()
plt.title(r"J/$\psi$ -> True, $B_0 \, mass \geq 5350$ -> False")
plt.savefig("Output/B0_mm_XGB_filter_Jpsi_mass.pdf")
plt.show()
#%%