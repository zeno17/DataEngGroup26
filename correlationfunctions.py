# -*- coding: utf-8 -*-
"""
Created on Sat May  9 21:28:45 2020

@author: Zeno
"""

import numpy as np

def pearsonrcustom(x, y):
    # Custom function based on scipy framework
    x = np.asarray(x)
    y = np.asarray(y)
 
    xmean = np.mean(x)
    ymean = np.mean(y)
    
    xm = x - xmean
    ym = y - ymean

    normxm = np.linalg.norm(xm)
    normym = np.linalg.norm(ym)
    r = np.dot(xm/normxm, ym/normym)

    #Ensure that correlation is absolute between 1 because of flop accuracy
    r = max(min(r, 1.0), -1.0)
    return r

def spearmanrcustom(x, y):
    ranksx = ranking(x)
    ranksy = ranking(y)
    r = pearsonrcustom(ranksx, ranksy)
    return r

def ranking(x):
    arr = np.ravel(np.asarray(x))
    sorter = np.argsort(arr, kind='mergesort')

    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size, dtype=np.intp)

    arr = arr[sorter]
    obs = np.r_[True, arr[1:] != arr[:-1]]
    dense = obs.cumsum()[inv]
    # cumulative counts of each unique value
    count = np.r_[np.nonzero(obs)[0], len(obs)]

    # average method
    return .5 * (count[dense] + count[dense - 1] + 1)  