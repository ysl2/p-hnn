#include <vector>

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (prefetch_current_) {
    prefetch_free_.push(prefetch_current_);
  }
  prefetch_current_ = prefetch_full_.pop("Waiting for data");
  // Reshape to loaded data.
  top[0]->ReshapeLike(prefetch_current_->data_);
  top[0]->set_gpu_data(prefetch_current_->data_.mutable_gpu_data());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(prefetch_current_->label_);
    top[1]->set_gpu_data(prefetch_current_->label_.mutable_gpu_data());
  }
}


template <typename Dtype>
void BasePrefetchingLabelmapDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  if (prefetch_current_) {
    prefetch_free_.push(prefetch_current_);
  }
  prefetch_current_ = prefetch_full_.pop("Waiting for data");

  // Reshape to loaded data.
  top[0]->ReshapeLike(prefetch_current_->data_);
  // Copy the data
  top[0]->set_gpu_data(prefetch_current_->data_.mutable_gpu_data());
  CHECK(labelmap_number_+1==top.size() ) << "Number of labels must equal top outputs + 1.";
  for (int j=0;j<labelmap_number_;++j){
    // Reshape to loaded labels.
    top[j+1]->ReshapeLike(*(prefetch_current_->labelmaps_[j]));
    // Copy the labels.

    top[j+1]->set_gpu_data(prefetch_current_->labelmaps_[j]->mutable_cpu_data());
  }

}


INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);
INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingLabelmapDataLayer);

}  // namespace caffe
