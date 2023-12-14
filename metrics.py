#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 12:49:04 2023

@author: maltejensen
"""

def FleissKappa(*args):
    '''
        Takes a variable number of predictions and one hot encodes and sum the votes, and return the
        fleiss Kappa. Works only for 2 classes.
        
        *args:      Sequence of scans to calculate the Fleiss Kappa for
    '''
    def one_hot(X):
        tmp = np.zeros(X.shape + (2,), dtype=np.int8)
        tmp[X == 0, 0] = 1 
        tmp[X == 1, 1] = 1 
        return tmp
    
    def aggregate_annotators(*args):
        agg_out = np.zeros_like(one_hot(args[0]), dtype=np.int8)
        for X in args:
            agg_out += one_hot(X)
    
        return agg_out
        
    agg_out = aggregate_annotators(*args)
    kappa = fleiss_kappa(agg_out.reshape(-1,2))

    return kappa