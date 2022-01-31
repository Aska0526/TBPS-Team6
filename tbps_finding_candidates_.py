# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 15:34:32 2022

Still messy and trying different things out
not good at coding... trying


"""
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
#%%  
# specifying the path to csv files
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
#ref:https://www.geeksforgeeks.org/read-multiple-csv-files-into-separate-dataframes-in-python/
#%%
td=df_list[1]
amc=df_list[0]

sig=df_list[2]
kbg=[]
for i in (3,4,5,6):
    kbg.append(df_list[i])
Ksmass=[]
for i in kbg:
    Ksmass.append(i['Kstar_MM'])

jpsi=[]
for i in(7,8,9,10):
    jpsi.append(df_list[i])

jpsim=[]
for i in jpsi:
    jpsim.append(i['J_psi_MM'])  
#sig=pd.read_csv('acceptance_mc.csv')
#pkmm=pd.read_csv('k_pi_swap.csv')



#%%td chisqaured 
MM=td['B0_MM']
chi_list=[]
"""
chi_name=[ 'B0_ENDVERTEX_CHI2', 'B0_FDCHI2_OWNPV',

     'Kstar_ENDVERTEX_CHI2', 'Kstar_FDCHI2_OWNPV', 

    'J_psi_ENDVERTEX_CHI2', 

      'J_psi_FDCHI2_OWNPV', 'B0_IPCHI2_OWNPV','Pi_IPCHI2_OWNPV']
"""
chi_name=[ 'B0_IPCHI2_OWNPV','Pi_IPCHI2_OWNPV', 'B0_FDCHI2_OWNPV']# focus on main chi^2 values

ndof=['B0_ENDVERTEX_NDOF','J_psi_ENDVERTEX_NDOF', 'Kstar_ENDVERTEX_NDOF']
#B0- 5
# Jpsi, Kstar-1
dof=[]
for i in ndof:
    temp=td[i]
    dof.append(temp)

for i in chi_name:
    temp=td[i]
    chi_list.append(temp)
    
#%% prob dist, check
prob=[  'mu_plus_MC15TuneV1_ProbNNmu']
plt.hist(td.mu_plus_MC15TuneV1_ProbNNmu,bins=1000,histtype=u'step')
#%% used them initially to see distribution
Chifd=td['B0_FDCHI2_OWNPV']
Chiip=td['B0_IPCHI2_OWNPV']
JpsiMM=td['J_psi_MM']
q2=td['q2']
Kstar=td['Kstar_MM']
dira=td['B0_DIRA_OWNPV']
B0MM=td['B0_MM']
MMsig=sig['B0_MM']
kpsMM=kps['Kstar_MM']
#%% K plots bg sources
#d=amc[(amc.q2 <4.5)&(amc.q2>1.5)]
listn=[]
for i in (0,1,2,3):

    listn.append(list_of_names[i+3])
    plt.hist(Ksmass[i],bins=500,histtype=u'step')
    #plt.hist(kbg[i].B0_MM,bins=500,histtype=u'step')
listn.append('td_kstar')
listn.append('sig_kstar')
plt.hist(td.Kstar_MM,bins=500,histtype=u'step')
plt.hist(sig.Kstar_MM,bins=500,histtype=u'step')
plt.legend(listn)
plt.title('Kstar -- possible backgrounds')
plt.xlabel('K*_MM/MeV')
plt.ylabel('Number')
#%%temp plot - to quickly check plots...
plt.hist(sig.J_psi_MM,bins=500,histtype=u'step')
plt.title('siganl Jpsi')


#%%jpsi bakcground sources
listn=[]
for i in (0,1,2,3):

    listn.append(list_of_names[i+7])
    plt.hist(jpsim[i],bins=500,histtype=u'step',density=True)
    #plt.hist(kbg[i].B0_MM,bins=500,histtype=u'step')
listn.append('td_jpsi')
listn.append('sig_jpsi')
plt.hist(td.J_psi_MM,bins=500,histtype=u'step',density=True)
plt.hist(sig.B0_MM,bins=500,histtype=u'step',density=True)
plt.legend(listn,loc='best',fontsize=8)
plt.title('Jpsi -- possible backgrounds')
plt.xlabel('Jpsi_MM/MeV')
plt.ylabel('Number')
plt.xlim(3000,4000)
#%% scatter plot
td.plot.scatter(x='B0_MM',y='q2',s=0.01)
plt.plot([5340,5340], [0,25], linestyle='solid', linewidth='1',color='b')
plt.plot( [5150,5700],[8,8], linestyle='dashed', linewidth='1',color='r')
plt.plot([5150,5700],[11,11] , linestyle='dashed', linewidth='1',color='r')
plt.plot([5150,5700],[12.5,12.5], linestyle='dashed', linewidth='1',color='k')
plt.plot([5150,5700], [15,15], linestyle='dashed', linewidth='1',color='k')
plt.plot([5220,5220], [0,25], linestyle='solid', linewidth='1',color='b')
plt.xlim(5179,5700)
plt.xlabel(' B0_MM/MeV')
plt.ylabel('q2/GeV')
plt.title('distribution')

#%% histrogram for B0 mass
plt.hist(td_filter10.B0_MM,bins=1000,histtype=u'step')
plt.xlabel('B0_MM/MeV')
plt.ylabel('number')
plt.title('B0 mass distribution')
#%%chi sqaured plot

for i in chi_list:
    plt.hist(i,bins=1000,histtype=u'step')
    plt.title(str(i.name))
    plt.show()


#%%chi sqaure test... tried different sig levels, might get them into one list 
crit_list30=[1.07,1.07,1.07]
crit_lis=[0.7,0.7,0.7]
crit_list10=[2.71,2.71,2.71]
crit_list=[10,10,10]
crit_list5=[3.84,3.84,3.84]
crit_list25=[5.024,5.024,5.024]
crit_list01=[6.635,6.635,6.635]
#%% chi filter
def chi_filter(df,lis):
    for i in chi_name:
        nam=df[i].name
        print(nam)
        ind=chi_name.index(i)
        df=df[(df[nam] >lis[ind])]
        #df_filter2=df_filter1[df_filter1.mu_plus_MC15TuneV1_ProbNNmu>0.95]
    return df



#%% total filter, some commented out (q2 / mass/ prob criterion)
def filt(df,lis):
    #df1=df[(df.J_psi_MM <3090)|(df.J_psi_MM >)&(df.B0_MM<5230)]
    #|(df.J_psi_MM<3200))
    #|(df.B0_MM>5350))]
    
    #df = df.drop(index=df1.index)
    #df=df[(df.J_psi_MM <3690)|(df.J_psi_MM>3700)]
    #df=df[(df.q2 >1.1)|(df.q2<0.98)]
    #df=df[(df.q2 >15)|(df.q2<12.5)]
    #df=df[(df.q2 >10)|(df.q2<9)]\ 
    for i in chi_name:
        nam=df[i].name
       # print(nam)
        ind=chi_name.index(i)
        df=df[(df[nam] >lis[ind])]
    df=df[(df.Kstar_MM>800)]
    #df_filter2=df_filter1[df_filter1.mu_plus_MC15TuneV1_ProbNNmu>0.95]
    return df

td_filter30=filt(td,crit_list30)
name=td_filter30
plt.hist(name.B0_MM,bins=1000,histtype=u'step')
plt.hist(td.B0_MM,bins=1000,histtype=u'step')
#%%full filt.
td_filter5=filt(td,crit_list5)
amc_filter30=filt(amc,crit_list30)
#%%chi only 
td_filter30chi=chi_filter(td,crit_list30)
td_filter10=chi_filter(td,crit_list10)
td_filter5=chi_filter(td,crit_list5)

#%%
amc_5=chi_filter(amc,crit_list5)
amc_10=chi_filter(amc,crit_list10)
amc_30=chi_filter(amc,crit_list30)

#%%
tdlist=[td_filter5,td_filter30,td_filter10]
amclist=[amc_5,amc_10,amc_30]
amcname=['amc_5','a_10',"a_30"]
name=['td_5','td_10',"td_30"]
#%% plot
name=td_filter30
plt.hist(name.B0_MM,bins=1000,histtype=u'step')
plt.xlabel('B0 MM signal/MeV')
plt.ylabel('number')
plt.title('td filter(Jpsi;X2;K*) number left is'+ str(name.shape[0]))


plt.hist(td.B0_MM,bins=1000,histtype=u'step')
#plt.title('td filter(X^2) number left is'+ str(name.shape[0]))
#%%
for i in (0,1,2):
    tdlist[i].to_csv(name[i]+'.csv')

#%%

for i in (0,1,2):
    amclist[i].to_csv(amcname[i]+'.csv')
#%%
for i in amclist:
    
    plt.hist(i.B0_MM,bins=1000,histtype=u'step')
    plt.xlabel('B0 MM signal/MeV')
    plt.ylabel('number')
    #plt.title('amc chi_filetr number left is'+ str(i.shape[0]))

    
plt.hist(amc.B0_MM,bins=1000,histtype=u'step')
plt.xlabel('B0_MM/MeV')
plt.ylabel('number')

#plt.title('B0 mass distribution')
    
#%%
for i in tdlist:
    
    plt.hist(i.B0_MM,bins=1000,histtype=u'step')
    plt.xlabel('B0 MM signal/MeV')
    plt.ylabel('number')
    plt.title('td chi_filetr number left is'+ str(i.shape[0]))
    plt.show()

    
    
plt.hist(td.B0_MM,bins=1000,histtype=u'step')
plt.xlabel('B0_MM/MeV')
plt.ylabel('number')
#%%
plt.hist(MM,bins=1000,histtype=u'step')
plt.xlabel('B0_MM/MeV')
plt.ylabel('number')
plt.title('B0 mass distribution')
#%%
plt.legend(['all 6 chi2 filtered','original'])


    
#%% Fprming list of filtered B0 mass, with desired values
# I guess this part about B0 filter is on hold, range could not be too strict, turned to focus on backgrounds)
B_name=['B1','B2']
B_bounds=[[5866,4779],[5500,5000]]
B_filter=[]
for i in B_bounds:
    temp_df=td[(td.B0_MM <i[0])&(td.B0_MM>i[1])]
    temp_list=[]
    temp_list.append(i[0])
    temp_list.append(i[1])
    temp_list.append(temp_df)
    B_filter.append(temp_list)

#%%forming list of removed data after mass filtering for background analysis
B_removed=[]
for i in B_bounds:
    temp_df=td[(td.B0_MM >i[0])|(td.B0_MM<i[1])]
    temp_list=[]
    temp_list.append(i[0])
    temp_list.append(i[1])
    temp_list.append(temp_df)
    B_removed.append(temp_list)

#%% Chi-squared filter

np.mean(Chifd)#8493
np.median(Chifd)#1758
bound_chifd= ...
chifd_filtered=td[(td.B0_FDCHI2_OWNPV<bound_chifd)]
#%% CHI squared B0 IP filter
np.mean(Chiip)#3.273
np.median(Chiip)#1.9715

chifd_filtered=td[(td.B0_IPCHI2_OWNPV>0.9)]

name=chifd_filtered.B0_MM
plt.hist(name,bins=1000,histtype=u'step')
plt.xlabel('B0 MM signal/MeV')
plt.ylabel('number')
plt.title(str(name.name)+' - no filtering applied')


#%% peaking BG:B0 to K* J/psi decay mode- Jpsi rest mass 3096
#consider q^2 and calculated J/psi

Jpsi_filter=td[(td.B0_MM <3176)&(td.J_psi_MM>2946)]



#%% peaking BG: misreading K pi and K+ K-
#calculate K pi mass and campare with phi mass
#and if pion is kaon like ?
#phi- 1019.46
#%% CHI^2 just to see...
name=Chiip
plt.hist(name,bins=1000)
plt.xlabel(str(name.name))
plt.ylabel('Number of candiates')
plt.title(str(name.name)+' - no filtering applied')
#%%
name=Chifd
plt.hist(name,bins=1000)
plt.xlabel(str(name.name))
plt.ylabel('Number of candiates')
plt.title(str(name.name)+' - no filtering applied')
