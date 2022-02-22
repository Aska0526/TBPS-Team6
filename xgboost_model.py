#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
#%%
TEST_DATA_PATH = "Output_Data/dataset_manual_filter_1.pkl"
df_test = pd.read_pickle(TEST_DATA_PATH)

TOTAL_DATA_PATH = "Data/total_dataset.pkl"
df_total = pd.read_pickle(TOTAL_DATA_PATH)

# SIGNAL_DATA_PATH = "Data/signal.pkl"
SIGNAL_DATA_PATH = "Data/jpsi.pkl"
df_signal = pd.read_pickle(SIGNAL_DATA_PATH)

dfs = [df_test, df_total, df_signal]

df_test["B0_PT"] = df_test["mu_minus_PT"] + df_test["mu_plus_PT"] + df_test["K_PT"] + df_test["Pi_PT"]
df_test["B0_PX"] = df_test["mu_minus_PX"] + df_test["mu_plus_PX"] + df_test["K_PX"] + df_test["Pi_PX"]
df_test["B0_PY"] = df_test["mu_minus_PY"] + df_test["mu_plus_PY"] + df_test["K_PY"] + df_test["Pi_PY"]
df_test["B0_PZ"] = df_test["mu_minus_PZ"] + df_test["mu_plus_PZ"] + df_test["K_PZ"] + df_test["Pi_PZ"]
df_test["B0_PE"] = df_test["mu_minus_PE"] + df_test["mu_plus_PE"] + df_test["K_PE"] + df_test["Pi_PE"]


df_total["B0_PT"] = df_total["mu_minus_PT"] + df_total["mu_plus_PT"] + df_total["K_PT"] + df_total["Pi_PT"]
df_total["B0_PX"] = df_total["mu_minus_PX"] + df_total["mu_plus_PX"] + df_total["K_PX"] + df_total["Pi_PX"]
df_total["B0_PY"] = df_total["mu_minus_PY"] + df_total["mu_plus_PY"] + df_total["K_PY"] + df_total["Pi_PY"]
df_total["B0_PZ"] = df_total["mu_minus_PZ"] + df_total["mu_plus_PZ"] + df_total["K_PZ"] + df_total["Pi_PZ"]
df_total["B0_PE"] = df_total["mu_minus_PE"] + df_total["mu_plus_PE"] + df_total["K_PE"] + df_total["Pi_PE"]

df_signal["B0_PT"] = df_signal["mu_minus_PT"] + df_signal["mu_plus_PT"] + df_signal["K_PT"] + df_signal["Pi_PT"]
df_signal["B0_PX"] = df_signal["mu_minus_PX"] + df_signal["mu_plus_PX"] + df_signal["K_PX"] + df_signal["Pi_PX"]
df_signal["B0_PY"] = df_signal["mu_minus_PY"] + df_signal["mu_plus_PY"] + df_signal["K_PY"] + df_signal["Pi_PY"]
df_signal["B0_PZ"] = df_signal["mu_minus_PZ"] + df_signal["mu_plus_PZ"] + df_signal["K_PZ"] + df_signal["Pi_PZ"]
df_signal["B0_PE"] = df_signal["mu_minus_PE"] + df_signal["mu_plus_PE"] + df_signal["K_PE"] + df_signal["Pi_PE"]

#%%
drop_columns = ["year", "B0_MM"]
df_test_xgb = df_test.drop(columns=drop_columns)
#%%
mass_filter = df_total['B0_MM'] >= 5350

df_background = df_total[mass_filter]

df_signal = df_signal.sample(n=len(df_background), random_state=6)

#%%
df_signal["B0_PT"] = df_signal["mu_minus_PT"] + df_signal["mu_plus_PT"] + df_signal["K_PT"] + df_signal["Pi_PT"]
df_background["B0_PT"] = df_background["mu_minus_PT"] + df_signal["mu_plus_PT"] + df_signal["K_PT"] + df_signal["Pi_PT"]
#%%
df_signal["signal"] = np.ones(len(df_signal), dtype=bool)
df_background["signal"] = np.zeros(len(df_background), dtype=bool)

df_train = pd.concat([df_signal, df_background], ignore_index=True)
df_train = df_train.drop(columns=drop_columns)
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
max_depth = 15
xg_clf = xgb.XGBClassifier(n_estimators=300, verbosity=1, eta=0.1, max_depth=max_depth)
#%%
xg_clf.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_validate, y_validate)], early_stopping_rounds=10)
#%%
# xg_clf.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_validate, y_validate)], xgb_model=xg_clf.get_booster())

#%%
y_pred = xg_clf.predict(x_validate)
print(accuracy_score(y_validate, y_pred))

disp_train = ConfusionMatrixDisplay.from_estimator(
        xg_clf,
        x_train,
        y_train,
        display_labels=["Background", "Signal"],
        cmap=plt.cm.Blues,
        normalize="true",
    )
plt.title(f"Training Dataset, Depth = {max_depth}")
plt.savefig(f"Output/confusion_training_XGB_filter_Jpsi_mass_depth_{max_depth}_equal.pdf")
plt.savefig(f"Output/confusion_training_XGB_filter_Jpsi_mass_depth_{max_depth}_equal.png", dpi=1000)
plt.show()

disp_validate = ConfusionMatrixDisplay.from_estimator(
        xg_clf,
        x_validate,
        y_validate,
        display_labels=["Background", "Signal"],
        cmap=plt.cm.Blues,
        normalize="true",
    )
plt.title(f"Validation Dataset, Depth = {max_depth}")
plt.savefig(f"Output/confusion_validate_XGB_filter_Jpsi_mass_depth_{max_depth}_equal.pdf")
plt.savefig(f"Output/confusion_validate_XGB_filter_Jpsi_mass_depth_{max_depth}_equal.png", dpi=1000)
plt.show()
#%%
df_clf_filtered_xgb = df_test_xgb.loc[xg_clf.predict(df_test_xgb), :]
df_clf_filtered = df_test.loc[df_clf_filtered_xgb.index, :]
#%%
plt.hist(df_test["B0_MM"], bins=100, label="Manual Cuts only")
plt.hist(df_clf_filtered["B0_MM"], bins=100, label="XG Boosted Decision Tree")
plt.xlabel("B0 mass / MeV")
plt.ylabel("Number of Candidates")
plt.legend()
plt.title(f"#Signal = #Background, B0 Momenta, Max Depth = {max_depth}")
# plt.title(r"J/$\psi$ -> True, $B_0 \, mass \geq 5350$ -> False, #Signal = #Background")
plt.savefig(f"Output/B0_mm_XGB_filter_Jpsi_mass_depth_{max_depth}_equal.pdf")
plt.savefig(f"Output/B0_mm_XGB_filter_Jpsi_mass_depth_{max_depth}_equal.png", dpi=1000)
plt.show()
#%%
plt.title(f"#Signal = #Background, B0 Momenta, Max Depth = {max_depth}")
plt.hist(df_clf_filtered["costhetal"], bins=50)
plt.xlabel(r"$cos(\theta_l)$")
plt.ylabel("Number of Candidates")
plt.savefig(f"Output/costhetal_XGB_filter_Jpsi_mass_depth_{max_depth}_equal.pdf")
plt.savefig(f"Output/costhetal_XGB_filter_Jpsi_mass_depth_{max_depth}_equal.png", dpi=1000)
plt.show()
#%%
plt.title(f"#Signal = #Background, B0 Momenta, Max Depth = {max_depth}")
plt.hist(df_clf_filtered["costhetak"], bins=50)
plt.xlabel(r"$cos(\theta_k)$")
plt.ylabel("Number of Candidates")
plt.savefig(f"Output/costhetak_XGB_filter_Jpsi_mass_depth_{max_depth}_equal.pdf")
plt.savefig(f"Output/costhetak_XGB_filter_Jpsi_mass_depth_{max_depth}_equal.png", dpi=1000)
plt.show()
#%%
training_loss = xg_clf.evals_result()["validation_0"]["logloss"]

plt.plot(np.arange(len(training_loss)), training_loss)
plt.yscale("log")
plt.show()
#%%