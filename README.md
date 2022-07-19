
Code and scripts to run progressive holistically-nested networks (P-HNN). Model files and scripts are provided for pathological lung segmentation, but model can be trained and applied for any segmentation task, be it natural image or medical image. 

### PyTorch Version

The model has been converted from its old caffe version to pytorch. You can find the old caffe version and documentation in [caffe_version](/caffe_version). Conversion was performed by adapting the [pytorch-caffe](https://github.com/marvis/pytorch-caffe) repository. You can find the adapted version in [pytorch_caffe](/pytorch_caffe). Many thanks to the author of that repository. 


To run the new version, you need to have [PyTorch](https://pytorch.org/), [caffe](http://caffe.berkeleyvision.org/), and [nibabel](https://nipy.org/nibabel/) installed. For caffe, I recommend just installing it using the Anaconda environment, **but make sure you use the default channel**, and not from any other channel, e.g., conda_forge, because there are too many incompatibilities across channels for caffe:
>conda install caffe

There is no need to install caffe-gpu, caffe is only needed to load the model. 

Apart from that, the operation of P-HNN is very similar to the old version. 

### Running pre-trained model

To get a probability map:

>python predict\_phnn\_pytorch.py --file\_in <any_nibabel_compatible_file,.hdr,.ni,.ni.gz> --file\_out <pmap_out> --batch\_size <number_slices_to_run_in_batch>

If you run out of memory, you probably have to reduce the batch\_size.

To threshold probablity map into a binary map run:

>python threshold\_pmap.py --file\_in <pmap_in> --file\_out <binary_out>

This will use the default probability map threshold determined through validation tests. You can provide your own if you wish to override.

There are more argument options, which you can consult by running --help to either of the scripts, but the default options should get you started

Note that by default the model that is used is the one from fold_0, i.e., the 1st fold in our cross validated experiments. We also include a list of training and validation and test volumes from the LTRC and University Hospitals of Geneva datasets that we used for each fold. So if you're testing on CT scans from those same datasets, then you should make sure you aren't using a model trained on the same volume. If so, you should pick a model from a different cross validation fold. 


### Training

Although it has not been tested, it should be very possible to finetune/retrain PHNN in pytorch. Our model is adapted from [HED](https://github.com/s9xie/hed), which in turn is built off of the highly popular VGG-16 architecture.


### Citations

If you are using the code/model/data provided here in a publication, please cite our paper:


  @inproceedings{Harrison\_2017,
	  author    = {Adam P. Harrison and Ziyue Xu and Kevin George and Le Lu and Ronald M. Summers and Daniel J. Mollura},
	  title     = {Progressive and Multi-Path Holistically Nested Neural Networks for Pathological Lung Segmentation from CT Images},
	  booktitle = {{MICCAI} 2017},
	  pages     = {1--8},
	  year      = {2017},
  
}
