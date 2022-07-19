#include <vector>

#include "caffe/layers/threshold_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ThresholdForward(const int n, const Dtype threshold,
    const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > threshold ? in[index] : 0;
  }
}

template <typename Dtype>
void ThresholdLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ThresholdForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, threshold_, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void ThresholdBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, Dtype threshold_) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * (in_data[index] > threshold_);
  }
}

template <typename Dtype>
void ThresholdLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();

    // NOLINT_NEXT_LINE(whitespace/operators)
    ThresholdBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff, threshold_);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(ThresholdLayer);


}  // namespace caffe
