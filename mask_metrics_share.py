# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 09:46:51 2016

@author: adam
"""



def threshold_mask(v_data,threshold):
    import scipy.ndimage.morphology
    from skimage import measure
    import numpy as np
    v_data[v_data>threshold]=1
    v_data[v_data<1]=0
    all_labels = measure.label(v_data)
    props=measure.regionprops(all_labels)
    props.sort(key=lambda x:x.area,reverse=True) #sort connected components by area
    thresholded_mask=np.zeros(v_data.shape)
    if len(props)>=2:
        print props[0].area
        print props[1].area
        if props[0].area/props[1].area>5: #if the largest is way larger than the second largest
            thresholded_mask[all_labels==props[0].label]=1 #only turn on the largest component
        else:
            thresholded_mask[all_labels==props[0].label]=1 #turn on two largest components
            thresholded_mask[all_labels==props[1].label]=1 
    elif len(props):
        thresholded_mask[all_labels==props[0].label]=1
        
    thresholded_mask=scipy.ndimage.morphology.binary_fill_holes(thresholded_mask).astype(np.uint8)    
    return thresholded_mask


    