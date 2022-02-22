# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 17:24:57 2022

@author: jingyi
"""


#trying to be more organised @_@


import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from iminuit import Minuit
import glob
import statistics as stats
#%%
os.chdir(r'C:\Users\jingyi\OneDrive - Imperial College London\Desktop\year3-problem-solving')
#cwd=os.getcwd()
#%%  reading df
# specifying the psath to csv files
path = r"C:\Users\jingyi\OneDrive - Imperial College London\Desktop\year3-problem-solving"
  
  
# csv files in the path
files = glob.glob(path + "/*.csv")
  
# assign dataset names
list_of_names = ['acceptance_mc','total_dataset','signal','phimumu','pKmumu_piTok_kTop','pKmumu_piTop','k_pi_swap','jpsi_mu_pi_swap','jpsi_mu_k_swap',
                 'psi2S','jpsi']
 
# create empty list
df_list = []

# append datasets into the list
for i in range(len(list_of_names)):
    temp_df = pd.read_csv(path+"/"+list_of_names[i]+".csv")
    df_list.append(temp_df)
    
############################################################################################### 
#%% dataset
td=df_list[2]
#%% filter general
def filte1(df,crit,variable,boo): #1 variable
    if boo==True:
        for i in range (0, len(crit)):
             df=df[(df[variable[i]]<crit[i])]
        return df
    if boo==False:
        for i in range (0, len(crit)):
             df=df[(df[variable[i]]>crit[i])]
        return df


def filte2(df,crit,variable,boo):
    """
    Df=dataframe
    Crit= boundary pairs lists
    2 boundaries, e.g.: crit=[[lower(smaller) bound, upper bound],[lb2,ub2],[lb3,ub3]]
    Variable= what you are filtering
    Boo=True/false
    """
    if boo==True:#keep those within range
        for i in range (0, len(crit)):
             df=df[(df[variable[i]]<crit[i][1])&(df[variable[i]]>crit[i][0])]
        return df
    if boo==False:#keep those outside range
        for i in range (0, len(crit)):
             df[(df[variable[i]]>crit[i][1])|(df[variable[i]]<crit[i][0])]
        return df

        
#%%  
      
def plots(df,namelist,title):
    """
    plots the mass and angular distributions of a dataset, superposed onthe original td distributions
    """
    for i in namelist:
        plt.hist(df[i],bins=1000,histtype=u'step')
        plt.title(i+'  number left  '+ str(df.shape[0])+title)
        plt.hist(td[i],bins=1000,histtype=u'step')
        plt.legend([i+' filtered','original td distribution'])
        plt.show()
anglemass=['B0_MM','costhetal','costhetak','q2','phi']
plots(td,anglemass,'td')

        
#%% examine distribution - initial
anglemass=['B0_MM','costhetal','costhetak','q2','phi']
plots(td,anglemass,'td distribution')

############################################################################################### 
#%% k pi swap removing pi very likely to be K

tdk=filte1(td,[0.95],['Pi_MC15TuneV1_ProbNNk'],True)
plots(tdk,anglemass,'td removing pi likely to be k')


#%% pion likely to be proton

tdpip=filte1(td,[0.50],['Pi_MC15TuneV1_ProbNNp'],True)
plots(tdpip,anglemass,'td removing pi likely to be proton')


#%% chi^2
chi_name=[ 'Pi_IPCHI2_OWNPV', 'B0_FDCHI2_OWNPV']

#%% q2 filter example- for psi 2 S
tdq2=filte2(td,[[8,11]],['q2'],True)
plots(tdq2,anglemass,'td q2>11 or q2<8')