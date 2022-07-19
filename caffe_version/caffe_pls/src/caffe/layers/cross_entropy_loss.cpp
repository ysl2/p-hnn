#include <algorithm>
#include <vector>

#include "caffe/layers/cross_entropy_loss.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);

  if (this->layer_param_.loss_param().beta_size()){
    CHECK_EQ(this->layer_param_.loss_param().beta_size(), 1) <<
      "CROSS_ENTROPY_LOSS layer can only have one or zero beta values.";
  }
}


template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "CROSS_ENTROPY_LOSS layer inputs must have the same count.";


}




template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.


  const int count = bottom[0]->count();


  // Compute the loss (negative log likelihood)

  const int num = bottom[0]->num();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  Dtype loss = 0;

  float beta(0.5);
  if (this->layer_param_.loss_param().beta_size()){
    beta=this->layer_param_.loss_param().beta(0);


    for (int i = 0; i < count; ++i) {

      Dtype prob = target[i]*input_data[i]+(1-target[i])*(1-input_data[i]);
      prob=std::max(prob, Dtype(kLOG_THRESHOLD));
      loss -= log(prob)*((target[i])*(1-2*beta)+beta);

    }


  }
  else{


    for (int i = 0; i < count; ++i) {

      Dtype prob = target[i]*input_data[i]+(1-target[i])*(1-input_data[i]);
      prob=std::max(prob, Dtype(kLOG_THRESHOLD));
      loss -= log(prob);

    }

  }
  top[0]->mutable_cpu_data()[0] = loss / num;

}

template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }

  if (propagate_down[0]) {

    const int num = bottom[0]->num();
    const Dtype loss_weight = top[0]->cpu_diff()[0]/num;
    const int count = bottom[0]->count();
    const Dtype* input_data = bottom[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    Dtype* target = bottom[1]->mutable_cpu_data();
    float beta(0.5);
    if (this->layer_param_.loss_param().beta_size()){

      CHECK_EQ(this->layer_param_.loss_param().beta_size(), 1) <<
        "CROSS_ENTROPY_LOSS layer can only have one or zero beta values.";
      beta=this->layer_param_.loss_param().beta(0);         // First, compute the diff



      for (int i = 0; i < count; ++i) {
        Dtype prob= -target[i]*std::max( input_data[i], Dtype(kLOG_THRESHOLD))+(1-target[i])*std::max((1-input_data[i]),Dtype(kLOG_THRESHOLD));

        bottom_diff[i] = ((target[i])*(1-2*beta)+beta)*loss_weight / prob;
      }

    }
    else{


      for (int i = 0; i < count; ++i) {
        Dtype prob= -target[i]*std::max( input_data[i], Dtype(kLOG_THRESHOLD))+(1-target[i])*std::max((1-input_data[i]),Dtype(kLOG_THRESHOLD));
        bottom_diff[i] = loss_weight / prob;
      }

    }
  }
}

#ifdef CPU_ONLY
#STUB_GPU_BACKWARD(CrossEntropyLossLayer, Backward);
#endif

INSTANTIATE_CLASS(CrossEntropyLossLayer);
REGISTER_LAYER_CLASS(CrossEntropyLoss);

}  // namespace caffe
