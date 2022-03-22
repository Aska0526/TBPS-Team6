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
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from xgboost import cv
from sklearn.inspection import permutation_importance
#%%
TEST_DATA_PATH = "Output_Data/dataset_manual_filter_3.pkl"
df_test = pd.read_pickle(TEST_DATA_PATH)

TOTAL_DATA_PATH = "Data/total_dataset.pkl"
df_total = pd.read_pickle(TOTAL_DATA_PATH)

jpsi_mu_k_swap_DATA_PATH = "Data/jpsi_mu_k_swap.pkl"
df_jpsi_mu_k_swap = pd.read_pickle(jpsi_mu_k_swap_DATA_PATH)

jpsi_mu_pi_swap_DATA_PATH = "Data/jpsi_mu_pi_swap.pkl"
df_jpsi_mu_pi_swap = pd.read_pickle(jpsi_mu_pi_swap_DATA_PATH)

k_pi_swap_DATA_PATH = "Data/k_pi_swap.pkl"
df_k_pi_swap = pd.read_pickle(k_pi_swap_DATA_PATH)

phimumu_DATA_PATH = "Data/phimumu.pkl"
df_phimumu = pd.read_pickle(phimumu_DATA_PATH)

pKmumu_piTok_kTop_DATA_PATH = "Data/pKmumu_piTok_kTop.pkl"
df_pKmumu_piTok_kTop = pd.read_pickle(pKmumu_piTok_kTop_DATA_PATH)

pKmumu_piTop_DATA_PATH = "Data/pKmumu_piTop.pkl"
df_pKmumu_piTop = pd.read_pickle(pKmumu_piTop_DATA_PATH)


SIGNAL_DATA_PATH = "Data/signal.pkl"
# SIGNAL_DATA_PATH = "Data/jpsi.pkl"
df_signal = pd.read_pickle(SIGNAL_DATA_PATH)

def add_columns(df):
    df["B0_PT"] = df["mu_minus_PT"] + df["mu_plus_PT"] + df["K_PT"] + df["Pi_PT"]
    df["B0_PX"] = df["mu_minus_PX"] + df["mu_plus_PX"] + df["K_PX"] + df["Pi_PX"]
    df["B0_PY"] = df["mu_minus_PY"] + df["mu_plus_PY"] + df["K_PY"] + df["Pi_PY"]
    df["B0_PZ"] = df["mu_minus_PZ"] + df["mu_plus_PZ"] + df["K_PZ"] + df["Pi_PZ"]
    df["B0_PE"] = df["mu_minus_PE"] + df["mu_plus_PE"] + df["K_PE"] + df["Pi_PE"]
    return df

df_test = add_columns(df_test)
df_total = add_columns(df_total)
df_jpsi_mu_k_swap = add_columns(df_jpsi_mu_k_swap)
df_jpsi_mu_pi_swap = add_columns(df_jpsi_mu_pi_swap)
df_k_pi_swap = add_columns(df_k_pi_swap)
df_phimumu = add_columns(df_phimumu)
df_pKmumu_piTok_kTop = add_columns(df_pKmumu_piTok_kTop)
df_pKmumu_piTop = add_columns(df_pKmumu_piTop)
df_signal = add_columns(df_signal)

#%%
drop_columns = ["year", "B0_MM"]
df_test_xgb = df_test.drop(columns=drop_columns)
#%%
mass_filter = df_total['B0_MM'] >= 5350

df_background_small = pd.concat(
    [
        df_jpsi_mu_k_swap,
        df_jpsi_mu_pi_swap,
        df_pKmumu_piTok_kTop,
        df_pKmumu_piTop,
    ]
)

df_background_large = pd.concat(
    [
        df_total[mass_filter],
        df_k_pi_swap,
        df_phimumu,
    ]
)

df_background = pd.concat(
    [
        df_background_small.sample(n=len(df_background_large), random_state=6, replace=True),
        df_background_large
    ]
)

#%%
if len(df_background) > len(df_signal):
    df_signal = df_signal.sample(n=len(df_background), random_state=6, replace=True)
elif len(df_signal) > len(df_background):
    df_background = df_background.sample(n=len(df_signal), random_state=6, replace=True)

#%%
df_signal["signal"] = np.ones(len(df_signal), dtype=bool)
df_background["signal"] = np.zeros(len(df_background), dtype=bool)

df_train = pd.concat([df_signal, df_background], ignore_index=True)
df_train = df_train.drop(columns=drop_columns)
#%%
x_train, x_validate, y_train, y_validate = train_test_split(
    df_train[df_train.columns[~df_train.columns.isin(["signal"])]], 
    df_train["signal"],
    test_size=0.2,
    random_state=6)

#%%
max_depth = 12
xg_clf = xgb.XGBClassifier(n_estimators=300, verbosity=1, eta=0.1, max_depth=max_depth)

xg_clf.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_validate, y_validate)], early_stopping_rounds=10)
#%%
# xg_clf.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_validate, y_validate)], xgb_model=xg_clf.get_booster(), early_stopping_rounds=10)

#%%
# xg_clf.save_model(f"Model/signal_td_reconstruction_balanced_oversampling_treedepth_{max_depth}_eta_0.1_state_6.model")

bst = xgb.XGBClassifier()
bst.load_model("Model/signal_td_reconstruction_balanced_oversampling_treedepth_12_eta_0.1_state_6.model")
xg_clf=bst
#%%

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
plt.title(f"Training dataset")
# plt.savefig(f"Output/confusion_training_XGB_filter_signal_td_reconstruction_depth_{max_depth}_equal_oversampling.png", dpi=1000)
plt.show()

disp_validate = ConfusionMatrixDisplay.from_estimator(
        xg_clf,
        x_validate,
        y_validate,
        display_labels=["Background", "Signal"],
        cmap=plt.cm.Blues,
        normalize="true",
    )
plt.title(f"Validation dataset")
# plt.savefig(f"Output/confusion_validation_XGB_filter_signal_td_reconstruction_depth_{max_depth}_equal_oversampling.png", dpi=1000)
plt.show()

#%%
# xg_clf = xgb.XGBClassifier()
# xg_clf.load_model("Model/signal_td_balanced_oversampling_treedepth_15_eta_0.1_state_6.model")

# xg_clf.get_booster().feature_names = list(x_train.columns)

importance_gain = xg_clf.get_booster().get_score(importance_type="gain")
importance_gain_sorted = sorted(importance_gain.items(), key=lambda x: x[1], reverse=True)

num_features = 15
importance_properties = [importance_gain_sorted[i][0] for i in range(num_features)]
importance_score = [importance_gain_sorted[i][1] for i in range(num_features)]

plt.barh(importance_properties[::-1], importance_score[::-1])
# plt.yticks(np.arange(num_features - 1, -1, -1), importance_properties)
plt.show()
#%%
perm_importance = permutation_importance(xg_clf, x_validate, y_validate, n_repeats=10, random_state=6)
#%%
sorted_idx = perm_importance.importances_mean.argsort()

fig, ax = plt.subplots(nrows = 1, ncols = 1)
ax.barh(np.array(x_train.columns)[sorted_idx][-15:], perm_importance.importances_mean[sorted_idx][-15:])
plt.xlabel("Permutation Importance")
plt.title("Feature Importance")
plt.tight_layout()
# plt.savefig(f"Output/permutation_importance_XGB_filter_signal_td_reconstruction_depth_{max_depth}_equal_oversampling.png", dpi=1000)
plt.show()
#%%
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (18,8))
xgb.plot_importance(xg_clf, ax=ax, max_num_features=10, grid=False, importance_type="gain", height=0.3, title=f"Feature Importance, Depth = {max_depth}")
# xgb.plot_importance(xg_clf, ax=ax, max_num_features=20, grid=False, height=0.3, title=f"Feature Importance, Depth = 15")
# plt.savefig(f"Output/importance_XGB_filter_signal_td_reconstruction_depth_{max_depth}_equal_oversampling.jpeg", dpi=1000)
# plt.savefig("Output/importance_XGB_filter_signal_td_depth_15_equal_oversampling2.png", dpi=1000)
plt.show()

#%%
df_clf_filtered_xgb = df_test_xgb.loc[xg_clf.predict(df_test_xgb), :]
df_clf_filtered = df_test.loc[df_clf_filtered_xgb.index, :]

# df_clf_filtered.to_pickle(f"Output_Data/XGB_filter_signal_td_noq2_depth_{max_depth}_equal_oversampling.pkl")
#%%
colors = plt.rcParams['axes.prop_cycle'].by_key()["color"]

plt.hist(df_test["B0_MM"], bins=60, label="Manual cuts only", color=colors[0])
plt.hist(df_clf_filtered["B0_MM"], bins=60, label="XGBoost Classifier", color=colors[1])
plt.xlabel(r"$B_0$ mass / MeV")
plt.ylabel("Number of candidates")
plt.legend()
# plt.title(f"signal.pkl + td, Oversampling Background, B0_P, Max Depth = {max_depth}")
plt.title(r"$B_0$ mass distribution")
# plt.savefig(f"Output/B0mm_XGB_filter_signal_td_reconstruction_depth_{max_depth}_equal_oversampling.png", dpi=1000)
plt.show()
#%%
plt.hist(df_test["B0_MM"], bins=60, label="Manual cuts only", density=True, alpha=1, color=colors[0])
plt.hist(df_clf_filtered["B0_MM"], bins=60, label="XGBoost Classifier", density=True, alpha=0.5, color=colors[1])
# plt.hist(df_test["B0_MM"], bins=60, histtype="step", label="Manual cuts only", density=True, color=colors[0])
plt.hist(df_signal["B0_MM"], bins=60, histtype="step", linewidth=1, label="Signal Monte Carlo", density=True, color=colors[2])
plt.xlabel(r"$B_0$ mass / MeV")
plt.ylabel("Probability density")
plt.legend()
# plt.title(f"signal.pkl + td, Oversampling Background, B0_P, Max Depth = {max_depth}")
plt.title(r"Normalised $B_0$ mass distribution")
# plt.savefig(f"Output/B0mm_normalised_XGB_filter_signal_td_reconstruction_depth_{max_depth}_equal_oversampling_v4.png", dpi=1000)
plt.show()
#%%
plt.title(f"signal.pkl + td, Oversampling Background, B0_P, Max Depth = {max_depth}")
plt.hist(df_clf_filtered["costhetal"], bins=50)
plt.xlabel(r"$cos(\theta_l)$")
plt.ylabel("Number of Candidates")
# plt.savefig(f"Output/costhetal_XGB_filter_signal_td_noq2_depth_{max_depth}_equal_oversampling.png", dpi=1000)
plt.show()
#%%
plt.title(f"signal.pkl + td, Oversampling Background, B0_P, Max Depth = {max_depth}")
plt.hist(df_clf_filtered["costhetak"], bins=50)
plt.xlabel(r"$cos(\theta_k)$")
plt.ylabel("Number of Candidates")
# plt.savefig(f"Output/costhetak_XGB_filter_signal_td_noq2_depth_{max_depth}_equal_oversampling.png", dpi=1000)
plt.show()
#%%
training_loss = xg_clf.evals_result()["validation_0"]["logloss"]

plt.plot(np.arange(len(training_loss)), training_loss)
plt.yscale("log")
plt.show()
#%%
bins_b = [0.1, 1.1, 2.5, 4, 6, 15, 17, 11, 1, 15]

bins_t = [0.98, 2.5, 4, 6, 8, 17, 19, 12.5, 6, 17.9]

q = [[],[],[],[],[],[], [],[],[],[]]

d = [[],[],[],[],[],[], [],[],[],[]]

df = [[],[],[],[],[],[], [],[],[],[]]

ds = df_test

ds_filtered = df_clf_filtered

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

    ax.hist(d[i]["B0_MM"], bins=100)

    ax.hist(df[i]["B0_MM"],bins = 100)

    ax.title.set_text(titles[i])

    ax.set_ylabel("Number of Candidates")

    ax.set_xlabel("B0 mass / MeV")

    #ax.set_ylim(0, 150)

# plt.savefig(f"Output/B0mm_q2bins_XGB_filter_signal_td_noq2_depth_{max_depth}_equal_oversampling.png", dpi=1000)
plt.show()
#%%
# fit_params = {
#     "eval_set" : [(x_train, y_train), (x_validate, y_validate)],
#     # "early_stopping_rounds" : 10
# }
# results = cross_val_score(xg_clf, x, y, cv=kfold, verbose=1, fit_params=fit_params)

# params = {
#     "objective" : "binary:logistic",
#     'learning_rate' : 0.3,
#     'max_depth' : 12
#     }

# xgb_cv = cv(dtrain=dmatrix, params=params, nfold=10, num_boost_round=3, early_stopping_rounds=10, seed=6)
#%%