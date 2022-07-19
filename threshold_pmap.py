#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 14:13:44 2017

@author: Adam P Harrison
adam.p.harrison@gmail.com

Progressive and Multi-Path Holistically Nested Neural Networks for Pathological Lung Segmentation from CT Images
"""

if __name__ == "__main__": 
    import argparse
    
    import mask_metrics_share as m
    import nibabel as nib
    import numpy as np
    
    parser = argparse.ArgumentParser(description='Threshold a P-HNN Proability Map.')    
   
    parser.add_argument('--file_in', type=str, required=True,help='path to input probability map .hdr file')
    parser.add_argument('--file_out', type=str, required=True,help='path to output .hdr file')    
    parser.add_argument('--threshold', type=int, required=False,default=0.78,help='threshold for probability map, should be between 0 and 1s') 
    
    
    args=parser.parse_args()
    volume=nib.load(args.file_in)
    v_data=np.squeeze(volume.get_data())
    result=m.threshold_mask(v_data,args.threshold)
    
    result=nib.Nifti1Image(result,volume.get_affine(),volume.get_header())     
    nib.save(result,args.file_out)