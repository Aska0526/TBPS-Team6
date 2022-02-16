# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 15:43:29 2022

@author: 范朝
"""

#Details of the logic >>> look for teams logbook

import numpy as np

def get_sig(edges, A, mu, bckgrd):
    '''
    Params:
        edges: list/array, x-coordinates of the data returned from histogram
        A: float64, gaussian signal amplitude returned from curvefit popt
        mu: float64, gaussian signal mean returned from curvefit popt
        bckgrd: function, bakground function that returns 
                            a value for background with a given x-coord
                            
    Returns: 
        Sig float64, a number representing the 
                    significance of this signal, used to compare between files
                    
    '''
    xprime = edges - mu
    index = np.argmin(xprime) 
    h = bckgrd(edges[index])

    Sig = A/h
    return Sig