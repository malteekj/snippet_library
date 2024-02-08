#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 23:31:19 2024

@author: maltejensen
"""

import scipy
import numpy as np

def slicedDilationOrErosion(input_mask, num_iteration, operation):
    '''
    Perform the dilation on the smallest slice that will fit the
    segmentation
    '''
    margin = 2 if num_iteration is None else num_iteration+1
    
    # find the minimum volume enclosing the organ
    x_idx = np.where(input_mask.sum(axis=(1,2)))[0]
    x_start, x_end = x_idx[0]-margin, x_idx[-1]+margin
    y_idx = np.where(input_mask.sum(axis=(0,2)))[0]
    y_start, y_end = y_idx[0]-margin, y_idx[-1]+margin
    z_idx = np.where(input_mask.sum(axis=(0,1)))[0]
    z_start, z_end = z_idx[0]-margin, z_idx[-1]+margin
    
    struct = scipy.ndimage.generate_binary_structure(3,1)
    struct = scipy.ndimage.iterate_structure(struct, num_iteration)
    
    if operation == 'dilate':
        mask_slice = scipy.ndimage.binary_dilation(input_mask[x_start:x_end, y_start:y_end, z_start:z_end], structure=struct).astype(np.int8)
    elif operation == 'erode':
        mask_slice = scipy.ndimage.binary_erosion(input_mask[x_start:x_end, y_start:y_end, z_start:z_end], structure=struct).astype(np.int8)
        
    output_mask = input_mask.copy()
    
    output_mask[x_start:x_end, y_start:y_end, z_start:z_end] = mask_slice
    
    return output_mask

