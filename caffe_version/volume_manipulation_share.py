# -*- coding: utf-8 -*-
"""
Created on Thu May 12 15:30:38 2016

@author: adam
"""
import multiprocessing
import os
import nibabel as nib

import numpy as np
import errno


def windowSlice(slice, min_val, max_val):
    slice[slice < min_val] = min_val
    slice[slice > max_val] = max_val
    slice -= min_val
    slice = slice/(max_val-min_val)
    slice = slice*255
    return slice


def makeImToColour(slice, min_vals, max_vals):

    colour_image = np.zeros(slice.shape+(3,))
    colour_image[:, :, 0] = windowSlice(slice, min_vals[0], max_vals[0])
    colour_image[:, :, 1] = windowSlice(slice, min_vals[1], max_vals[1])
    colour_image[:, :, 2] = windowSlice(slice, min_vals[2], max_vals[2])
    return colour_image


def makeSliceToColour(volume, slice_num, min_vals, max_vals):

    slice = np.squeeze(volume[:, :, slice_num]).astype(np.double)

    colour_image = np.zeros(slice.shape+(3,))
    colour_image[:, :, 0] = windowSlice(slice, min_vals[0], max_vals[0])
    colour_image[:, :, 1] = windowSlice(slice, min_vals[1], max_vals[1])
    colour_image[:, :, 2] = windowSlice(slice, min_vals[2], max_vals[2])
    return colour_image


def makeSliceToColourWindow(volume, slice_num, ct_windows):

    min_vals = ct_windows[:, 0]-ct_windows[:, 1]/2
    max_vals = ct_windows[:, 0]+ct_windows[:, 1]/2
    return makeSliceToColour(volume, slice_num, min_vals, max_vals)




def fillmask(mask, do_slices=False):
    import scipy.ndimage.morphology
    mask = scipy.ndimage.morphology.binary_fill_holes(mask).astype(np.uint8)
    if do_slices:
        dims = mask.shape

        for i in range(0, dims[2]):
            mask[:, :, i] = scipy.ndimage.morphology.binary_fill_holes(
                mask[:, :, i]).astype(np.uint8)

    return mask



def rw_background(volume):
    import skimage.segmentation as seg
    import numpy as np

    back_mask = np.zeros_like(volume)

    back_mask[volume > 200] = 2
    back_mask = fillmask(back_mask, True)
    back_mask = 1 - back_mask
    return back_mask
