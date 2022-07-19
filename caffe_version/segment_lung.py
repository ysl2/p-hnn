#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:46:28 2017

@author: Adam P Harrison
adam.p.harrison@gmail.com

Progressive and Multi-Path Holistically Nested Neural Networks for Pathological Lung Segmentation from CT Images
"""


def run_lung_segmentation(file_in, file_out, mean_file, offset, gpu, batch_size, caffe_root, model_weights, model_spec):
    import load_hed_snapshot_share as l
    import nibabel as nib
    import numpy as np

    ct_windows = np.array([[600, 1200], [1040, 400], [225, 450]])
    net = l.load_hed_net(model_spec, model_weights, caffe_root)
    volume = nib.load(file_in)
    v_data = np.squeeze(volume.get_data())
    mean_image = np.load(mean_file)

    result_data = l.run_volume_hed(
        net, v_data, mean_image, ct_windows, offset, gpu, batch_size, caffe_root)

    result = nib.Nifti1Image(
        result_data, volume.get_affine(), volume.get_header())
    nib.save(result, file_out)


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(description='Run P-HNN on hdr image.')

    parser.add_argument('--gpu', type=int, default=0, help='gpu device to use')
    parser.add_argument('--caffe_root', type=str,
                        required=True, help='location of caffe build')
    parser.add_argument('--file_in', type=str, required=True,
                        help='path to input .hdr file')
    parser.add_argument('--file_out', type=str, required=True,
                        help='path to output .hdr file')
    parser.add_argument('--offset', type=int, required=False, default=1024,
                        help='offset to add to attenuation values (ONLY CHANGE IF YOU KNOW WHAT YOU\'RE DOING')
    parser.add_argument('--mean_file', type=str, required=False,
                        default='', help='location of mean file to use')
    parser.add_argument('--batch_size', type=int, required=False, default=5,
                        help='batch number of slices; higher number makes segmentation go faster, but requires more memory')
    parser.add_argument('--model_weights', type=str,
                        required=False, default='', help='Path to model file')
    parser.add_argument('--model_spec', type=str, required=False,
                        default='', help='Path to model prototxt')

    args = parser.parse_args()
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    print( cur_dir)
    mean_file = args.mean_file
    if mean_file == '':
        mean_file = os.path.join(
            cur_dir, 'caffe_model/train_colour_slice_list_0.lst_mean_image.npy')

    model_weights = args.model_weights
    if model_weights == '':
        model_weights = os.path.join(
            cur_dir, 'caffe_model/fold_0/hed_iter_13000.caffemodel')

    model_spec = args.model_spec
    if model_spec == '':
        model_spec = os.path.join(
            cur_dir, 'caffe_model/fold_0/deploy.prototxt')

    run_lung_segmentation(args.file_in, args.file_out, mean_file, args.offset,
                          args.gpu, args.batch_size, args.caffe_root, model_weights, model_spec)
