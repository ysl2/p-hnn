#include <algorithm>
#include <vector>

#include "caffe/layers/sigmoid_cross_entropy_loss_layer.hpp"

#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);

  if (this->layer_param_.loss_param().beta_size()){
    CHECK_EQ(this->layer_param_.loss_param().beta_size(), 1) <<
      "SIGMOID_CROSS_ENTROPY_LOSS layer can only have one or zero beta values.";
  }


  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (this->layer_param_.loss_param().has_normalization()) {
    normalization_ = this->layer_param_.loss_param().normalization();
  } else if (this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = LossParameter_NormalizationMode_BATCH_SIZE;

  }

  beta_=0.5;
  is_beta_=false;
  if (this->layer_param_.loss_param().beta_size()){

      CHECK_EQ(this->layer_param_.loss_param().beta_size(), 1) <<
        "SIGMOID_CROSS_ENTROPY_LOSS layer can only have one or zero beta values.";
      beta_=this->layer_param_.loss_param().beta(0);
      is_beta_=true;
  }
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  outer_num_ = bottom[0]->shape(0);  // batch size
  inner_num_ = bottom[0]->count(1);  // instance size: |output| == |target|
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

// TODO(shelhamer) loss normalization should be pulled up into LossLayer,
// instead of duplicated here and in SoftMaxWithLossLayer
template <typename Dtype>
Dtype SigmoidCrossEntropyLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(outer_num_ * inner_num_);
      } else {
        normalizer = Dtype(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);

  const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
  Dtype tmax(-1e20);
  Dtype tmin(1e20);

  const int count = bottom[0]->count();


  // Compute the loss (negative log likelihood)



  // Compute the loss (negative log likelihood)

  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  int valid_count = 0;
  Dtype loss = 0;



  if (is_beta_){





    for (int i = 0; i < count; ++i) {
      const int target_value = static_cast<int>(target[i]);
      if (has_ignore_label_ && target_value == ignore_label_) {
        continue;
      }
      loss-= ((target_value)*(1-2*beta_)+beta_)*(input_data[i] * (target_value - (input_data[i] >= 0)) -
            log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0))));
      ++valid_count;
    }



  }
  else{


    for (int i = 0; i < count; ++i) {
      const int target_value = static_cast<int>(target[i]);
      if (has_ignore_label_ && target_value == ignore_label_) {
        continue;
      }

      loss -= input_data[i] * (target_value - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));

      ++valid_count;

    }

  }
  normalizer_ = get_normalizer(normalization_, valid_count);
  top[0]->mutable_cpu_data()[0] = loss / normalizer_;

}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }

  if (propagate_down[0]) {
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();

    Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer_;
    const int count = bottom[0]->count();


    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    if (is_beta_){
      Dtype* target = bottom[1]->mutable_cpu_data();
          // First, compute the diff


      caffe_sub(count, sigmoid_output_data, target, bottom_diff);
      // Scale down gradient, incorporating the class weight beta
      if (has_ignore_label_) {
        for (int i = 0; i < count; ++i) {
          const int target_value = static_cast<int>(target[i]);
          if (target_value == ignore_label_) {
            bottom_diff[i] = 0;
          }
          else{
            bottom_diff[i]*=loss_weight*(beta_+target_value*(1-2*beta_));
          }
        }
      }
      else{
        for (int i = 0; i < count; ++i) {
          bottom_diff[i]*=loss_weight*(beta_+static_cast<int>(target[i])*(1-2*beta_));

        }
      }

//      caffe_scal(count,loss_weight*(1-2*beta)/num,target);
//      caffe_add_scalar(count,loss_weight*beta/num,target);
//      caffe_mul(count, target,bottom_diff, bottom_diff);

    }
    else{

      const Dtype* target = bottom[1]->cpu_data();
      caffe_sub(count, sigmoid_output_data, target, bottom_diff);
      // Scale down gradient
      if (has_ignore_label_) {
        for (int i = 0; i < count; ++i) {
          const int target_value = static_cast<int>(target[i]);
          if (target_value == ignore_label_) {
            bottom_diff[i] = 0;
          }
        }
      }
      caffe_scal(count, loss_weight , bottom_diff);

    }

  }
}

#ifdef CPU_ONLY
STUB_GPU(SigmoidCrossEntropyLossLayer);
#endif

INSTANTIATE_CLASS(SigmoidCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(SigmoidCrossEntropyLoss);

}  // namespace caffe
