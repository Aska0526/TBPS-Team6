# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 02:46:34 2022

@author: jingyi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
os.chdir()
#%%








def filters(df):
    '''find acceptance function'''

    # choose candidates with one muon PT > 3.5GeV 2018 DOI:10.1007/JHEP10(2018)047
    PT_mu_filter = df['mu_minus_PT'] >= 3.5*(10**3)
    # DOI:10.1007/JHEP10(2018)047
    PT_K_filter = (df['K_PT'] +df['Pi_PT'])>= 3*(10**3)
    # Physics Letters B 753 (2016) 424–448
    PT_B0_filter =( df['mu_plus_PT']+df['mu_minus_PT']+df['K_PT']+df['Pi_PT']) >= 8*(10**3)
# Selected B0_IPCHI2<9  (3 sigma) 
    IP_B0_filter = df['B0_IPCHI2_OWNPV'] < 9
 
        #1112.3515v3
    Kstarmass= (df['Kstar_MM'] <= 992) & (df['Kstar_MM'] >= 792)

# should be numerically similar to number of degrees of freedom for the decay (5) 
    end_vertex_chi2_filter = df['B0_ENDVERTEX_CHI2'] < 10

# At least one of the daughter particles should have IPCHI2>16 (4 sigma) 
    daughter_IP_chi2_filter = (df['mu_minus_IPCHI2_OWNPV'] >= 16) | (df['mu_plus_IPCHI2_OWNPV'] >= 16)
    
# B0 should travel about 10mm (Less sure about this one - maybe add an upper limit?) 
    flight_distance_B0_filter =(df['B0_FD_OWNPV'] <= 500) & (df['B0_FD_OWNPV'] >= 8)

# cos(DIRA) should be close to 1
    DIRA_angle_filter = df['B0_DIRA_OWNPV'] > 0.99999

# Remove J/psi peaking background
    Jpsi_filter = (df["q2"] <= 8) | (df["q2"] >= 11)

# Remove psi(2S) peaking background
    psi2S_filter = (df["q2"] <= 12.5) | (df["q2"] >= 15)

# Possible pollution from Bo -> K*0 psi(-> mu_plus mu_minus)
    phi_filter = (df['q2'] <= 0.98) | (df['q2'] >= 1.1)

# Pion likely to be kaon
    pi_to_be_K_filter = df["Pi_MC15TuneV1_ProbNNk"] < 0.95

# Kaon likely to be pion
    K_to_be_pi_filter = df["K_MC15TuneV1_ProbNNpi"] < 0.95

# pion likely to be proton
    pi_to_be_p_filter = df["Pi_MC15TuneV1_ProbNNp"] < 0.9


#Applying filters (you can remove any filter to play around with them)
    df_filtered = df[
        end_vertex_chi2_filter
        &daughter_IP_chi2_filter
        &flight_distance_B0_filter
        & DIRA_angle_filter
        & Kstarmass
        & Jpsi_filter
        & psi2S_filter
        & phi_filter
        &IP_B0_filter
    
       &PT_B0_filter
       &PT_mu_filter
       &PT_K_filter

        
        #&PT_K_filter
    & pi_to_be_K_filter
    & K_to_be_pi_filter
    & pi_to_be_p_filter
        ]
    
    return df_filtered




td= pd.read_pickle('total_dataset.pkl')
tdf=filters(td)
plt.hist(tdf.costhetal,bins=100)
plt.show()
plt.hist(tdf.B0_MM,bins=100)
