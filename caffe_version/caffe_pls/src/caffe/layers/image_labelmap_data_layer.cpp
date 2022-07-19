#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <cstdlib>
#include <boost/algorithm/string.hpp>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/image_labelmap_data_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
ImageLabelmapDataLayer<Dtype>::~ImageLabelmapDataLayer<Dtype>() {
  this->StopInternalThread();
}
struct both_slashes {
    bool operator()(char a, char b) const {
        return a == '/' && b == '/';
    }
  };
template <typename Dtype>
void ImageLabelmapDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();

  scale_gt_  = this->layer_param_.image_label_map_data_param().scale_gt();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;

  std::ifstream infile(source.c_str());
  std::string line;
  vector<string> fields;


  while (std::getline(infile, line))
  {
    boost::trim(line);

    boost::split(fields,line,boost::is_any_of(" "));
    CHECK((fields.size()==this->labelmap_number_+1)) << "Must have " <<  this->labelmap_number_+1 << " entries in line";
    lines_.push_back(fields);

  }

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.

  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_][0],
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_][0];
  vector<cv::Mat> cv_gts(this->labelmap_number_);
  for(int j=0;j<this->labelmap_number_;++j){
    //std::cerr<< "Last: " << *(lines_[lines_id_][j+1].end()-2) <<std::endl;
    cv_gts[j]=ReadImageToCVMat(root_folder + lines_[lines_id_][j+1],
                                    new_height, new_width, 0);
    CHECK(cv_gts[j].data) << "Could not load " << lines_[lines_id_][j+1];
  }


  //const int channels = cv_img.channels();
  const int height = cv_img.rows;
  const int width = cv_img.cols;

  const int gt_channels = cv_gts[0].channels();
  const int gt_height = cv_gts[0].rows;
  const int gt_width = cv_gts[0].cols;

  std::cerr << "DIMS:" << height << " " << " " << width << " " << gt_height << " " << gt_width << std::endl;

  CHECK((height == gt_height) && (width == gt_width)) << "groundtruth size != image size";
  CHECK(gt_channels == 1) << "GT image channel number should be 1";

  for(int j=1;j<this->labelmap_number_;++j){
    CHECK((cv_gts[j].rows == gt_height) && (cv_gts[j].cols == gt_width)) << "mismatch in groundtruth size";
    CHECK(cv_gts[j].channels() == 1) << "GT image channel number should be 1";

  }

  std::cerr << "DIMS:" << height << " " << " " << width << " " << gt_height << " " << gt_width << std::endl;


  if (new_height > 0 && new_width > 0) {
    cv::resize(cv_img, cv_img, cv::Size(new_width, new_height));
    for(int j=0;j<this->labelmap_number_;++j){
      cv::resize(cv_gts[j], cv_gts[j], cv::Size(new_width, new_height));
    }

  }
    // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";

  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  vector<vector<int> > top_shape_labelmaps(this->labelmap_number_);
  this->transformed_data_.Reshape(top_shape);
  top_shape[0] = batch_size; //now set top shape to include batch size
  this->transformed_labelmaps_.resize(this->labelmap_number_);
  for(int j=0;j<this->labelmap_number_;++j){
      top_shape_labelmaps[j]=this->data_transformer_->InferBlobShape(cv_gts[j]);

      this->transformed_labelmaps_[j].reset(new Blob<Dtype>());
      this->transformed_labelmaps_[j]->Reshape(top_shape_labelmaps[j]);
      top_shape_labelmaps[j][0] = batch_size; //now set top shape to include batch size
  }






  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);

    this->prefetch_[i]->labelmaps_.resize(this->labelmap_number_);
    for(int j=0;j<this->labelmap_number_;++j){
      this->prefetch_[i]->labelmaps_[j].reset(new Blob<Dtype>());
      this->prefetch_[i]->labelmaps_[j]->Reshape(top_shape_labelmaps[j]);
    }


  }


  top[0]->Reshape(top_shape);
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  for(int j=0;j<this->labelmap_number_;++j){
    top[j+1]->Reshape(top_shape_labelmaps[j]);
    LOG(INFO) << "output label size: " << top[j+1]->num() << ","
      << top[j+1]->channels() << "," << top[j+1]->height() << ","
      << top[j+1]->width();
  }



}

template <typename Dtype>
void ImageLabelmapDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void ImageLabelmapDataLayer<Dtype>::load_batch(LabelmapBatch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  for(int j=0;j<this->labelmap_number_;++j){
    CHECK(batch->labelmaps_[j]->count());
  }

  CHECK(this->transformed_data_.count());
  for(int j=0;j<this->labelmap_number_;++j){
    CHECK(this->transformed_labelmaps_[j]->count());
  }

  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();

  string root_folder = image_data_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_][0],
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_][0];
  vector<cv::Mat> cv_gts(this->labelmap_number_);
  for(int j=0;j<this->labelmap_number_;++j){
    cv_gts[j]=ReadImageToCVMat(root_folder + lines_[lines_id_][j+1],
                                    new_height, new_width, 0);
    CHECK(cv_gts[j].data) << "Could not load " << lines_[lines_id_][j+1];
  }


  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  vector<vector<int> > top_shape_labelmaps(this->labelmap_number_);
  this->transformed_data_.Reshape(top_shape);
  top_shape[0] = batch_size; //now set top shape to include batch size
  batch->data_.Reshape(top_shape);
  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  vector<Dtype*> prefetch_labelmaps(this->labelmap_number_);
  for(int j=0;j<this->labelmap_number_;++j){
      top_shape_labelmaps[j]=this->data_transformer_->InferBlobShape(cv_gts[j]);


      this->transformed_labelmaps_[j]->Reshape(top_shape_labelmaps[j]);
      top_shape_labelmaps[j][0] = batch_size; //now set top shape to include batch size
      batch->labelmaps_[j]->Reshape(top_shape_labelmaps[j]);
      prefetch_labelmaps[j]=batch->labelmaps_[j]->mutable_cpu_data();

  }

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);

    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_][0],0, 0, is_color);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_][0];
    vector<cv::Mat> cv_gts(this->labelmap_number_);
    for(int j=0;j<this->labelmap_number_;++j){
      cv_gts[j]=ReadImageToCVMat(root_folder + lines_[lines_id_][j+1],0, 0, 0);
      CHECK(cv_gts[j].data) << "Could not load " << lines_[lines_id_][j+1];
    }

    const int height = cv_img.rows;
    const int width = cv_img.cols;

    const int gt_channels = cv_gts[0].channels();
    const int gt_height = cv_gts[0].rows;
    const int gt_width = cv_gts[0].cols;


    CHECK((height == gt_height) && (width == gt_width)) << "groundtruth size != image size";
    CHECK(gt_channels == 1) << "GT image channel number should be 1";

    for(int j=1;j<this->labelmap_number_;++j){
      CHECK((cv_gts[j].rows == gt_height) && (cv_gts[j].cols == gt_width)) << "mismatch in groundtruth size";
      CHECK(cv_gts[j].channels() == 1) << "GT image channel number should be 1";

    }

    if (new_height > 0 && new_width > 0) {
        cv::resize(cv_img, cv_img, cv::Size(new_width, new_height));
        bool good_data_pointer=true;
        for(int j=0;j<this->labelmap_number_;++j){
          cv::resize(cv_gts[j], cv_gts[j], cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);
          if (cv_gts[j].data)
            good_data_pointer=false;
        }

    }


    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    vector<int> offset_gts(this->labelmap_number_);
    for(int j=0;j<this->labelmap_number_;++j){
      offset_gts[j]=batch->labelmaps_[j]->offset(item_id);
    }

    //CHECK(offset == offset_gt) << "fetching should be synchronized";
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    for(int j=0;j<this->labelmap_number_;++j){
      this->transformed_labelmaps_[j]->set_cpu_data(prefetch_labelmaps[j] + offset_gts[j]);
    }


    vector<cv::Mat> encoded_gts(this->labelmap_number_);
    //regression
    if (scale_gt_){
      for(int j=0;j<this->labelmap_number_;++j){
        encoded_gts[j] = cv_gts[j]/255;
      }

    }
    else{
      for(int j=0;j<this->labelmap_number_;++j){
        encoded_gts[j] = cv_gts[j];
      }

    }


//    cv::minMaxLoc( encoded_gt, &minVal, &maxVal, &minLoc, &maxLoc );
//
//    std::cout << "encoded min val : " << minVal << " max val: " << maxVal << std::endl;



    //this->data_transformer_->LocTransform(cv_img, &(this->transformed_data_), h_off, w_off, do_mirror);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_),encoded_gts[0], this->transformed_labelmaps_[0].get());



    trans_time += timer.MicroSeconds();

    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImageLabelmapDataLayer);
REGISTER_LAYER_CLASS(ImageLabelmapData);

}  // namespace caffe
