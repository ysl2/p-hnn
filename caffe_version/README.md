Code and scripts to run progressive holistically-nested networks (P-HNN). Model files and scripts are provided for pathological lung segmentation, but model can be trained and applied for any segmentation task, be it natural image or medical image. 

### Running pre-trained model

You must have a compiled version of caffe. You can compile the version in the caffe subfolder, which has some slight modifications, but for inference the standard caffe should actually be sufficient. You'll also need appropriate python packages, e.g., scipy, numpy, and nibabel.

To get a probability map:

>python segment\_lung.py --caffe\_root <ROOT_of_compiled_caffe> --file\_in <any_nibabel_compatible_file,.hdr,.ni,.ni.gz> --file\_out <pmap_out> --batch\_size <number_slices_to_run_in_batch>

If you run out of memory, you probably have to reduce the batch\_size.

To threshold probablity map into a binary map run:

>python threshold\_pmap.py --file\_in <pmap_in> --file\_out <binary_out>

This will use the default probability map threshold determined through validation tests. You can provide your own if you wish to override.

There are more argument options, which you can consult by running --help to either of the scripts, but the default options should get you started

Note that by default the model that is used is the one from fold_0, i.e., the 1st fold in our cross validated experiments. We also include a list of training and validation and test volumes from the LTRC and University Hospitals of Geneva datasets that we used for each fold. So if you're testing on CT scans from those same datasets, then you should make sure you aren't using a model trained on the same volume. If so, you should pick a model from a different cross validation fold. 


### Training

Training the P-HNN model on new data is pretty much the same as training any other [caffe](http://caffe.berkeleyvision.org/) model, except you'll need to compile the caffe code in the /caffe\_pls folder, which has some modifications to the vanilla caffe. You probably want to fine-tune from an ImageNet pretrained model, which is what we did in our paper. Our model prototxt and solver prototxt can be found in the /caffe\_model folder, which provides more details on the settings we used to train the P-HNN model for pathological lung segmentation. Knowledge of how to train caffe models is required. Our model is adapted from [HED](https://github.com/s9xie/hed), which in turn is built off of the highly popular VGG-16 architecture.

Two differences from the standard caffe repo:
1) Like the [HED](https://github.com/s9xie/hed) code, from which we adapted our modifications, there is a new layer called image\_labelmap\_layer, which takes a list of images and ground truth masks for training and validation. Unlike HED, masks are expected to be {0,1} instead of {0,255}. List of files should be a line for each image and mask pair, of the format:  <img_name> <space> <mask_name>
2) We use a global balancing weight, unlike the image-specific weight used by HED. To set this properly you should calculate the ratio of positive\_pixels to all\_pixels in your training set. The beta in the SigmoidCrossEntropyLoss layers in train\_val.prototxt should then be set to the calculated value.

If you want to recreate our training, you can train on the public LTRC and University Hospitals of Geneva datasets that we used in our paper. Unfortunately, the NIH infection dataset is not pubicly available. We include train, test, and validation volume names for each cross validation fold within their respective folder. 


### Citations

If you are using the code/model/data provided here in a publication, please cite our paper:


  @inproceedings{Harrison\_2017,
	  author    = {Adam P. Harrison and Ziyue Xu and Kevin George and Le Lu and Ronald M. Summers and Daniel J. Mollura},
	  title     = {Progressive and Multi-Path Holistically Nested Neural Networks for Pathological Lung Segmentation from CT Images},
	  booktitle = {{MICCAI} 2017},
	  pages     = {1--8},
	  year      = {2017},
  
}
