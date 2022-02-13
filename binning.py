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
    
#%% the upgraded version (using quick insert method)
def searchInsert(q2):
    """
    Parameters
    ----------
    q2 : float number
        the q^2 from the data file

    Returns
    -------
    nums_index : integer
        the integer can be further converted to the bin numbers. The convention
        follows:
            
        bin0 => index 1
        bin1 => index 4
        bin2 => index 5
        bin3 => index 6
        bin4 => index 7
        bin5 => index 11
        bin6 => index 12 + 13
        bin7 => index 9
        bin8 => index 3 + 4 + 5 + 6
        bin9 => index 11 + 12
        
        The convention will be down in bin_q2 function
    """
    nums = [0.1, 0.98, 1, 1.1, 2.5, 4.0, 6.0, 8.0, 11.0, 12.5, 15.0, 17.0, 
                17.9, 19.0]
    nums_index = 0
    while len(nums) >= 1:
        nums_len = len(nums)
        mid_index = nums_len // 2
        if nums[mid_index] == q2:
            nums_index += mid_index
            return nums_index
        elif nums[mid_index] > q2:
            nums = nums[:mid_index]
            if len(nums) == 0:
                return nums_index
        else:
            nums_index += (len(nums[:mid_index + 1]))
            nums = nums[mid_index + 1:]
            if len(nums) == 0:
                return nums_index

def bin_q2(df, varible):
    '''
    Parameters
    ----------
    df : pandas.DataFrome 
        The file of csv converts to panda.DataFrame.
        
    varible : string
        name of varibles need to matching with the q^2. e.g If the costhetal is
        needed, just input "costhetal"

    Returns
    -------
    result : dict
        It inludes the correlated varible in each bins. The varible can be accessed by
        result["bin#"](# is the number of bin).
        The keys of dictionary is "bin0","bin1","bin2" ... "bin9"
    '''
    q2 = df["q2"]
    value = df[varible]
    result = {"bin0":[],"bin1":[], "bin2":[], "bin3":[], "bin4":[], "bin5":[], "bin6":[], "bin7":[], "bin8":[], "bin9":[]}
    for i in range(len(q2)):
        idx = searchInsert(q2[i])
        if idx == 1:
            result["bin0"].append(value[i])
        elif idx == 4:
            result["bin1"].append(value[i])
            result["bin8"].append(value[i])
        elif idx == 5:
            result["bin2"].append(value[i])
            result["bin8"].append(value[i])
        elif idx == 6:
            result["bin3"].append(value[i])
            result["bin8"].append(value[i])
        elif idx == 7:
            result["bin4"].append(value[i])
        elif idx == 11:
            result["bin5"].append(value[i])
            result["bin9"].append(value[i])
        elif idx == 12:
            result["bin6"].append(value[i])
            result["bin9"].append(value[i])
        elif idx == 9:
            result["bin7"].append(value[i])
        elif idx == 3:
            result["bin8"].append(value[i])
        elif idx == 13:
            result["bin6"].append(value[i])
    return result
