# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 22:23:39 2022

@author: 范朝 Chao Fan
"""
'''
This is the routine for binning datas according to their q^2 values.
Boundary scheme chosen from
 https://github.com/mesmith75/ic-teach-kstmumu-public/tree/main/predictions
 
Output df only contains cos(theta)_l, can manually add more.

'''

bin_boundary = [[0.1,0.98],[1.1,2.5],[2.5,4.0],[4.0,6.0],[6.0,8.0],[15.0,17.0],
        [17.0,19.0],[11.0,12.5],[1.0,6.0],[15.0,17.9]] #boundaries for 10 bins

bins = [] #list of bin df

for i in range(10):
    a , b = bin_boundary[i]
    bin_ = []
    for j in range(len(df)):
        if a < df['q2'][j] < b:
            ctl = df['costhetal'][j]
            q2 = df['q2'][j]
            bin_.append([q2,ctl])
    bin_ = pd.DataFrame(bin_,columns=['q2','ctl'])
    bins.append(bin_)