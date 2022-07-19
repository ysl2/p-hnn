from __future__ import division
from copy import deepcopy
from pytorch_caffe.caffenet import CaffeNet
from skimage.transform import resize
import sys
import torch
import numpy as np

import matplotlib.pyplot as plt
# use grayscale output rather than a (potentially misleading) color heatmap
plt.rcParams['image.cmap'] = 'gray'


def load_hed_solver(model_snapshot, solver_snapshot, solver_prototxt):

    # init
    caffe.set_mode_gpu()
# caffe.set_mode_cpu()
    caffe.set_device(0)

    solver = caffe.get_solver(solver_prototxt)

    solver.restore(solver_snapshot)
    return solver
    solver.net.copy_from(model_snapshot)
    solver.test_nets[0].copy_from(model_snapshot)
    return solver

    #test=nib.load(folder_in + '/001102_ild_inverted.hdr')
    # /mnt/ccruby/groups/DMprj/MingchenProjects/DATA/LTRC/ILD/001102_ild_inverted.hdr')
    # data=test.get_data()

    # slice=data[:,:,200]

    # plt.imshow(slice)

    # print test.shape


def sigmoid(x):
    import math
    return 1 / (1 + math.exp(-x))


def load_hed_net(model_file, model_snapshot):

    net = CaffeNet(model_file)
    net.load_state_dict(torch.load(model_snapshot))

    net.cuda()
    net.eval()

    return net



def run_image_hed(net, im, all_outputs=False, stagger=False):
    im = im.astype(np.float32)
    im = torch.from_numpy(im)
    im = im.cuda()
    net.forward(im)

    if all_outputs:
        result = np.zeros((6,)+im.shape[1:])
        result[0, :] = np.squeeze(net.blobs['sigmoid-dsn1'].data)
        result[1, :] = np.squeeze(net.blobs['sigmoid-dsn2'].data)
        result[2, :] = np.squeeze(net.blobs['sigmoid-dsn3'].data)
        result[3, :] = np.squeeze(net.blobs['sigmoid-dsn4'].data)
        result[4, :] = np.squeeze(net.blobs['sigmoid-dsn5'].data)
        result[5, :] = np.squeeze(net.blobs['sigmoid-fuse'].data)
    else:
        if not stagger:
            result = np.squeeze(net.blobs['sigmoid-fuse'].data)

        else:
            result = np.squeeze(net.blobs['sigmoid-dsn5'].data)
    result = result.detach().cpu().numpy()
    return result

def preprocessing(vol_slice, mean_image):

    vol_slice = np.transpose(vol_slice, (2,0,1))
    vol_slice = resize(vol_slice, mean_image.shape, order=1, preserve_range=True)

    ret_vol_slice = deepcopy(vol_slice)
    ret_vol_slice[0,:] = vol_slice[2,:]
    ret_vol_slice[2,:] = vol_slice[0,:]
    ret_vol_slice -= mean_image

    # ret_vol_slice /= 255
    return ret_vol_slice

def run_volume_hed(net, volume, mean_image, ct_windows, offset, batch_size):
    import volume_manipulation_share as v
    import numpy as np
    torch.set_grad_enabled(False)

    # transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    # transformer.set_transpose('data', (2, 0, 1))
    # # transformer.set_raw_scale('data',255)
    # if isinstance(mean_image, np.ndarray):
        # transformer.set_mean('data', mean_image)

    if isinstance(ct_windows, np.ndarray):
        # swap channels from RGB to BGR
        min_vals = ct_windows[:, 0]-ct_windows[:, 1]/2
        max_vals = ct_windows[:, 0]+ct_windows[:, 1]/2
    dims = volume.shape
    print(dims)

    volume += offset
    volume[volume < 0] = 0
    mask = np.zeros(dims)
    im_data = np.zeros((batch_size,) + mean_image.shape)


    print( 'Segmenting {:d} slices with batch size of {:d}'.format(dims[2], batch_size))
    for i in range(0, dims[2], batch_size):
        max_range = min(dims[2], i+batch_size)

        if max_range < i+batch_size:
            im_data = np.zeros((max_range-i,) + mean_image.shape)
        for j in range(i, max_range):
            print( 'Slice {:d}'.format(j))
            if isinstance(ct_windows, np.ndarray):
                slice = v.makeSliceToColour(volume, j, min_vals, max_vals)

            else:
                slice = volume[:, :, j]

                slice = np.expand_dims(slice, axis=2)

                slice = np.tile(slice, (1, 1, 3))
            transformed_image = preprocessing(slice, mean_image)

            im_data[j-i, :, :, :] = transformed_image

        result = run_image_hed(net, im_data, False, False)
        if result.ndim == 2:
            mask[:, :, j] = resize(result, dims[0:2], preserve_range=True)
        else:
            for j in range(i, max_range):
                mask[:, :, j] = resize(result[j-i, :, :], dims[0:2], preserve_range=True)

    return mask


def run_on_volume_folder_walk(net, folder_in, folder_out_sub, mean_file, ct_windows, offset, gpu=0, batch_size=10, stagger=False, threshold=0):

    import os
    import nibabel as nib
    import mask_metrics as m

    print(batch_size)
    shape_ = list(net.blobs['data'].data.shape)
    shape_[0] = batch_size
    net.blobs['data'].reshape(*tuple(shape_))
    net.reshape()

    if isinstance(mean_file, str):
        mean_image = np.load(mean_file)
    else:
        mean_image = 0

    for root, dirs, files in os.walk(folder_in):
        for name in files:
            if name.endswith(".hdr") and not 'mask' in name:

                file_name = os.path.join(root, name)
                print('Running on ' + file_name)

                volume = nib.load(file_name)
                v_data = np.squeeze(volume.get_data())

                result_data = run_volume_hed(
                    net, v_data, mean_image, ct_windows, offset, gpu, batch_size, stagger)
                volume_name = name[0:name.find(".hdr")]
                folder_out = root+'/'+folder_out_sub
                if not os.path.exists(folder_out):
                    os.makedirs(folder_out)

                volume_out_file = folder_out+'/'+volume_name+"_hed_mask.hdr"
                result = nib.Nifti1Image(
                    result_data, volume.get_affine(), volume.get_header())
                nib.save(result, volume_out_file)
                if threshold > 0:
                    print('Thresholding ' + file_name)
                    result_data = m.threshold_mask(result_data, threshold)
                    volume_out_file = folder_out+'/'+volume_name+"_hed_mask_threshold.hdr"
                    result = nib.Nifti1Image(
                        result_data, volume.get_affine(), volume.get_header())
                    nib.save(result, volume_out_file)


#ct_windows=np.array([[600, 1200], [1040, 400],[225,450]])
def run_on_volume_folder(net, folder_in, folder_out, mean_file, ct_windows, offset, gpu=0, batch_size=10, stagger=False, change_bg=False):

    import os
    import nibabel as nib
    import volume_manipulation as v

    print(batch_size)
    shape_ = list(net.blobs['data'].data.shape)
    shape_[0] = batch_size
    net.blobs['data'].reshape(*tuple(shape_))
    net.reshape()

    if isinstance(mean_file, str):
        mean_image = np.load(mean_file)
    else:
        mean_image = 0

    files = os.listdir(folder_in)

    for file in files:
        if file.endswith(".hdr") and not 'mask' in file:
            print('Running on ' + file)

            volume = nib.load(folder_in+'/'+file)
            v_data = np.squeeze(volume.get_data())

            if change_bg:

                v_data += offset  # just do the offset here
                v_data[v_data < 0] = 0
                back_mask = v.rw_background(v_data)
                v_data[back_mask == 1] = 2000
                offset = 0
            result_data = run_volume_hed(
                net, v_data, mean_image, ct_windows, offset, gpu, batch_size, stagger)
            volume_name = file[0:file.find(".hdr")]
            volume_out_file = folder_out+'/'+volume_name+"_hed_mask.hdr"
            result = nib.Nifti1Image(
                result_data, volume.get_affine(), volume.get_header())
            nib.save(result, volume_out_file)
