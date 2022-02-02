#%% All Filters - NEED TO RUN IMPORTS AND DATA IMPORT CELLS BELOW FIRST 
td_af = td
td_af = td_af.drop(td_af[(td_af['mu_minus_PT'] < 1.7*(10**3)) | (td_af['mu_plus_PT'] < 1.7*(10**3))].index)
td_af = td_af.drop(td_af[(td_af['B0_FD_OWNPV'] < 0.5*(10**1))].index)
td_af = td_af.drop(td_af[(td_af['B0_IPCHI2_OWNPV'] > 9)].index)
td_af = td_af.drop(td_af[(td_af['B0_ENDVERTEX_CHI2'] > 6)].index)
td_af = td_af.drop(td_af[(td_af['mu_minus_IPCHI2_OWNPV'] < 16) | (td_af['mu_plus_IPCHI2_OWNPV'] < 16)].index)
td_af = td_af.drop(td_af[(td_af['B0_DIRA_OWNPV'] < 0.99999)].index)
td_af = td_af.drop(td_af[(td_af['q2'] < 11) & (td_af['q2'] > 8)].index)
td_af = td_af.drop(td_af[(td_af['q2'] < 15) & (td_af['q2'] > 12.5)].index)
td_Af = td_af.drop(td_af[(td_af['q2'] < 1.1) & (td_af['q2'] > 0.98)].index)
B0_MM_af=td_af['B0_MM']
plt.hist(B0_MM_af,bins=200)
plt.xlabel('B0_MM/MeV')
plt.ylabel('Number of candiates')
plt.title('Filtered for all criteria')
plt.show()
#%%
# Imports 
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
style.use('dark_background')
import numpy as np
from iminuit import Minuit
import glob
#%%
os.chdir(r'C:\Users\sophi\OneDrive\Documents\Physics\Year_3\TBPS\year3-problem-solving')
path = r"C:\Users\sophi\OneDrive\Documents\Physics\Year_3\TBPS\year3-problem-solving"
files = glob.glob(path + "/*.csv")
list_of_names = ['acceptance_mc','jpsi', 'jpsi_mu_k_swap','jpsi_mu_pi_swap','k_pi_swap','phimumu','pKmumu_piTok_kTop','pKmumu_piTop','psi2S','signal','total_dataset'
                 ]
df_list = []
for i in range(len(list_of_names)):
    temp_df = pd.read_csv(path+"/"+list_of_names[i]+".csv")
    df_list.append(temp_df)
#%%Inital Mass Histogram 
td=df_list[10]
B0_MM=td['B0_MM']
plt.hist(B0_MM,bins=1000)
plt.xlabel('B0_MM/MeV')
plt.ylabel('Number of candiates')
plt.title('Rough examination(no filtering applied')
plt.show()

#%% Histograms of Transverse Momentum 
mu_minus_PT=td['mu_minus_PT']
plt.hist(mu_minus_PT,bins=1000)

mu_plus_PT=td['mu_plus_PT']
plt.hist(mu_plus_PT,bins=1000)

K_PT=td['K_PT']
plt.hist(K_PT,bins=1000)

Pi_PT=td['Pi_PT']
plt.hist(Pi_PT,bins=1000)

plt.xlabel('Transverse Momentum (MeV)')
plt.ylabel('Number of candiates')
plt.xlim((0,10000))
plt.legend(['mu_minus', 'mu_plus', 'K', 'Pi'])
plt.axvline(x=1.7*(10**3))
plt.savefig('Transverse Momentum Cutoff Histogram')
plt.show()

#%% Filter so at least one Muon has a Transverse Momentum greaster than a threshold
td_PV = td
td_PV = td_PV.drop(td[(td['mu_minus_PT'] < 1.7*(10**3)) | (td['mu_plus_PT'] < 1.7*(10**3))].index)
B0_MM_PV=td_PV['B0_MM']
plt.hist(B0_MM,bins=1000)
plt.hist(B0_MM_PV,bins=1000)
plt.xlabel('B0_MM/MeV')
plt.ylabel('Number of candiates')
plt.title('Transverse Momentum Filtered')
plt.legend(['Not Filtered', 'Filtered'])
plt.savefig('B0 MM corrected for Transverse Momentum ')
plt.show()

#%% Histograms of Flight Distance for B0
B0_FD_OWNPV=td['B0_FD_OWNPV']
plt.hist(B0_FD_OWNPV,bins=1000)
plt.xlabel('Flight Distance (mm)')
plt.ylabel('Number of candiates')
plt.title('Flight Distance of B0')
plt.xlim((0,100))
plt.axvline(x=0.5*(10**1))
plt.savefig('FD for B0 Histogram')
plt.show()
# %% Filter so Travels a minimum distance
"""Should it travel at least 1cm / around 1cm or more???? + should include chi2??"""
td_FD = td
td_FD = td_FD.drop(td[(td['B0_FD_OWNPV'] < 0.5*(10**1))].index)
B0_MM_FD=td_FD['B0_MM']
plt.hist(B0_MM,bins=1000)
plt.hist(B0_MM_FD,bins=1000)
plt.xlabel('B0_MM/MeV')
plt.ylabel('Number of candiates')
plt.title('Flight Distance Filtered')
plt.legend(['Not Filtered', 'Filtered'])
plt.savefig('B0 MM corrected for FD for B0')
plt.show()

#%% Histograms of IPCHI2 for B0
#Should be small 
B0_IPCHI2_OWNPV=td['B0_IPCHI2_OWNPV']
plt.hist(B0_IPCHI2_OWNPV,bins=1000)
plt.xlabel('IPCHI2')
plt.ylabel('Number of candiates')
plt.title('IPCHI2')
#plt.xlim((0,100))
plt.axvline(x=9)
plt.savefig('IPCHI2 for B0 Histogram')
plt.show()

# %% Filter so IPCHI2 for B0 is less than 3 sig
td_IPCHI2_BO = td
td_IPCHI2_BO = td_IPCHI2_BO.drop(td[(td['B0_IPCHI2_OWNPV'] > 9)].index)
B0_MM_IPCHI2_B0=td_IPCHI2_BO['B0_MM']
plt.hist(B0_MM,bins=1000)
plt.hist(B0_MM_IPCHI2_B0,bins=1000)
plt.xlabel('B0_MM/MeV')
plt.ylabel('Number of candiates')
plt.title('IPCHI" of B0 Filtered')
plt.legend(['Not Filtered', 'Filtered'])
plt.savefig('B0 MM corrected for IPCHI2 for B0')
plt.show()

#%% Histograms of End Vertex CHI2 for B0
#Should be less than number of degrees of freedom 
B0_ENDVERTEX_CHI2=td['B0_ENDVERTEX_CHI2']
plt.hist(B0_ENDVERTEX_CHI2,bins=1000)
plt.xlabel('End Vertex CHI2')
plt.ylabel('Number of candiates')
plt.title('End Vertex CHI2')
#plt.xlim((0,100))
plt.axvline(x=6)
plt.savefig('End Vertex CHI2 for B0 Histogram')
plt.show()

# %% Filter so End Vertex CHI2 for B0 is less than 3 sig
td_END_VERTEX_CHI2_BO = td
td_END_VERTEX_CHI2_BO = td_END_VERTEX_CHI2_BO.drop(td[(td['B0_ENDVERTEX_CHI2'] > 6)].index)
B0_MM_END_VERTEX_CHI2_B0=td_END_VERTEX_CHI2_BO['B0_MM']
plt.hist(B0_MM,bins=1000)
plt.hist(B0_MM_END_VERTEX_CHI2_B0,bins=1000)
plt.xlabel('B0_MM/MeV')
plt.ylabel('Number of candiates')
plt.title('End Vertex CHI2 of B0 Filtered')
plt.legend(['Not Filtered', 'Filtered'])
plt.savefig('B0 MM corrected for End Vertex CHI2 of B0')
plt.show()


# %%All Filters - including those below
td_af = td
td_af = td_af.drop(td_af[(td_af['mu_minus_PT'] < 1.7*(10**3)) | (td_af['mu_plus_PT'] < 1.7*(10**3))].index)
td_af = td_af.drop(td_af[(td_af['B0_FD_OWNPV'] < 0.5*(10**1))].index)
td_af = td_af.drop(td_af[(td_af['B0_IPCHI2_OWNPV'] > 9)].index)
td_af = td_af.drop(td_af[(td_af['B0_ENDVERTEX_CHI2'] > 6)].index)
td_af = td_af.drop(td_af[(td_af['mu_minus_IPCHI2_OWNPV'] < 16) | (td_af['mu_plus_IPCHI2_OWNPV'] < 16)].index)
td_af = td_af.drop(td_af[(td_af['B0_DIRA_OWNPV'] < 0.99999)].index)
td_af = td_af.drop(td_af[(td_af['q2'] < 11) & (td_af['q2'] > 8)].index)
td_af = td_af.drop(td_af[(td_af['q2'] < 15) & (td_af['q2'] > 12.5)].index)
td_Af = td_af.drop(td_af[(td_af['q2'] < 1.1) & (td_af['q2'] > 0.98)].index)
B0_MM_af=td_af['B0_MM']
#plt.hist(B0_MM,bins=1000)
plt.hist(B0_MM_af,bins=200)
plt.xlabel('B0_MM/MeV')
plt.ylabel('Number of candiates')
plt.title('Filtered for all criteria')
#plt.legend(['Not Filtered', 'Filtered'])
plt.savefig('B0 MM Filtered for all criteria')
plt.show()


#%% Histograms of IPCHI2 for Daughter Particles 
mu_minus_IPCHI2=td['mu_minus_IPCHI2_OWNPV']
plt.hist(mu_minus_IPCHI2,bins=10000)

mu_plus_IPCHI2=td['mu_plus_IPCHI2_OWNPV']
plt.xlim((0,8000))
plt.hist(mu_plus_IPCHI2,bins=10000)

K_IPCHI2=td['K_IPCHI2_OWNPV']
plt.hist(K_IPCHI2,bins=10000)

Pi_IPCHI2=td['Pi_IPCHI2_OWNPV']
plt.hist(Pi_IPCHI2,bins=10000)

plt.xlabel('Impact Parameter CHI2')
plt.ylabel('Number of candiates')
plt.xlim((0,2000))
#plt.ylim((0,50000))
plt.legend(['mu_minus', 'mu_plus', 'K', 'Pi'])
plt.axvline(x=16)
plt.savefig('IPCHI2 for Daughter Particles Cutoff Histogram')
plt.show()


#%% Filter so at least one Muon has a IPCHI2 greaster than a threshold
td_IPCHI2_muon = td
td_IPCHI2_muon = td_IPCHI2_muon.drop(td[(td['mu_minus_IPCHI2_OWNPV'] < 16) | (td['mu_plus_IPCHI2_OWNPV'] < 16)].index)
B0_MM_IPCHI2_muon=td_PV['B0_MM']
plt.hist(B0_MM,bins=1000)
plt.hist(B0_MM_IPCHI2_muon,bins=1000)
plt.xlabel('B0_MM/MeV')
plt.ylabel('Number of candiates')
plt.title('IPCHI2 Muons Filtered')
plt.legend(['Not Filtered', 'Filtered'])
plt.savefig('B0 MM corrected for IPCHI2 of Daughter Particles')
plt.show()
#%%
B0_DIRA_OWNPV=td['B0_DIRA_OWNPV']
plt.hist(B0_DIRA_OWNPV,bins=1000)
plt.xlabel('cos(DIRA) of B0 Own PV (mm)')
plt.ylabel('Number of candiates')
plt.title('cos(DIRA) of B0')
#plt.xlim((0,100))
plt.axvline(x=0.99999)
plt.savefig('cos(DIRA) for B0 Histogram')
plt.show()
# %%
td_B0_DIRA_OWNPV = td
td_B0_DIRA_OWNPV = td_B0_DIRA_OWNPV.drop(td[(td['B0_DIRA_OWNPV'] < 0.99999)].index)
B0_MM_DIRA_OWNPV=td_B0_DIRA_OWNPV['B0_MM']
plt.hist(B0_MM,bins=1000)
plt.hist(B0_MM_DIRA_OWNPV,bins=1000)
plt.xlabel('B0_MM/MeV')
plt.ylabel('Number of candiates')
plt.title('DIRA angle of B0 Filtered')
plt.legend(['Not Filtered', 'Filtered'])
plt.savefig('B0 MM corrected for DIRA angle ')
plt.show()

# %% q2 stuff
new_df = td
new_df = new_df.drop(td[(td['q2'] < 11) & (td['q2'] > 8)].index)
new_df = new_df.drop(td[(td['q2'] < 15) & (td['q2'] > 12.5)].index)
new_df = new_df.drop(td[(td['q2'] < 1.1) & (td['q2'] > 0.98)].index)
B0_MM_q2=new_df['B0_MM']
#plt.hist(B0_MM,bins=1000)
plt.hist(B0_MM_q2,bins=1000)
plt.xlabel('B0_MM/MeV')
plt.ylabel('Number of candiates')
plt.title('q2 Filtered')
#plt.legend(['Not Filtered', 'Filtered'])
plt.savefig('B0 MM corrected for q2 ')
plt.show()

# %%more q2 stuff
q2 = td['q2']
plt.hist2d(B0_MM, q2, bins = (200, 200), cmap=plt.cm.plasma,  norm=matplotlib.colors.LogNorm())#vmin=0.9, vmax = 5 ))
plt.colorbar()
plt.xlabel('B0_MM/MeV')
plt.ylabel('q2')
plt.axhline(y=11)
plt.axhline(y=8)
plt.axhline(y=15)
plt.axhline(y=12.5)
plt.axhline(y=1.1)
plt.axhline(y=0.98)
plt.savefig('q2 corrections')
plt.show()

#%% Histograms of FD_CHI2 for B0
#Should be small 
B0_FDCHI2_OWNPV=td['B0_FDCHI2_OWNPV']
plt.hist(B0_FDCHI2_OWNPV,bins=21000)
plt.xlabel('FDCHI2')
plt.ylabel('Number of candiates')
plt.title('FDCHI2 Histogram')
plt.xlim((0,2000))
plt.axvline(x=1000)
plt.savefig('FDCHI2 for B0 Histogram')
plt.show()

# %% Filter so FDCHI2 for B0 is less than 3 sig
td_FDCHI2_BO = td
td_FDCHI2_BO = td_FDCHI2_BO.drop(td[(td['B0_FDCHI2_OWNPV'] > 1000)].index)
B0_MM_FDCHI2_B0=td_FDCHI2_BO['B0_MM']
plt.hist(B0_MM,bins=1000)
plt.hist(B0_MM_FDCHI2_B0,bins=1000)
plt.xlabel('B0_MM/MeV')
plt.ylabel('Number of candiates')
plt.title('FDCHI2 of B0 Filtered')
plt.legend(['Not Filtered', 'Filtered'])
plt.savefig('B0 MM corrected for FDCHI2 for B0')
plt.show()


# %%
