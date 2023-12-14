#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 12:49:04 2023

@author: maltejensen
"""
from typing import Iterable 
import numpy as np

def flat(items, return_array=True):
    '''
    Function that takes in any iterable with potential nested iterables and 
    coverts into one list (or array). Can only handle one layer of nested
    iterables
    '''
    out = []
    if not isinstance(items, Iterable):
        return items
    
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in x:
                out.append(sub_x)
        else:
            out.append(x)
    
    if return_array:
        return np.array(out)
    else:
        return out
    