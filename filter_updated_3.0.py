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
       #arxiv:1112.3515v3
    PT_mu_filter = (df['mu_minus_PT']>= 1.5*(10**3)) |(df['mu_plus_PT']>= 1.5*(10**3) )

    
    # DOI:10.1007/JHEP10(2018)047
    PT_K_filter = (df['K_PT'] >= 0.5*(10**3))
    PT_pi_filter = (df['Pi_PT'] >= 0.5*(10**3))
    # Physics Letters B 753 (2016) 424–448
    PT_B0_filter =( df['mu_plus_PT']+df['mu_minus_PT']+df['K_PT']+df['Pi_PT']) >= 8*(10**3)
# Selected B0_IPCHI2<9  (3 sigma) 
    IP_B0_filter = df['B0_IPCHI2_OWNPV'] < 9
 
        #arxiv:1112.3515v3
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
    pi_to_be_K_filter = df["Pi_MC15TuneV1_ProbNNk"] < 0.8

# Kaon likely to be pion
    K_to_be_pi_filter = df["K_MC15TuneV1_ProbNNpi"] < 0.8

# pion likely to be proton
    pi_to_be_p_filter = df["Pi_MC15TuneV1_ProbNNp"] < 0.8


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
       #&PT_mup_filter
       &PT_K_filter
       &PT_pi_filter

        
        #&PT_K_filter
   #& pi_to_be_K_filter
    #& K_to_be_pi_filter
   # & pi_to_be_p_filter
        ]
    
    return df_filtered


def bin7(df):
    Jpsi_filter = (df["q2"] <= 12.5) & (df["q2"] >= 11)
    ds1=df[Jpsi_filter]
    return ds1

td= pd.read_pickle('total_dataset.pkl')


td1= pd.read_pickle('signal.pkl')

    
def plots(df,namelist,title):
    for i in namelist:
        plt.hist(df[i],bins=10000,histtype=u'step',density=True)

        plt.title(i+'  number left  '+ str(df.shape[0])+title)
        #plt.hist(td[i],bins=1000,histtype=u'step',density=True)
        plt.legend([i+' filtered','original td distribution'])
        
        plt.show()
anglemass=['B0_MM','costhetal','costhetak','q2','phi']
anglemass=['B0_MM','q2','mu_minus_PT','mu_minus_IPCHI2_OWNPV']
#plots(td,anglemass,'td')
#plots(td_filter30,anglemass,'td_filter30')

tdf=filters(td)
tds=filters(td1)
b7=bin7(tdf)
plots(tdf,anglemass,'td_filtered')
