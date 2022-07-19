#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/flatten_layer.hpp"
#include "caffe/layers/fourier_threshold_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template<typename TypeParam>
class FourierThresholdLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  FourierThresholdLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 5)),
        blob_top_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~FourierThresholdLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(FourierThresholdLayerTest, TestFloatAndDevices);

TYPED_TEST(FourierThresholdLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  FourierThresholdLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_top_->num_axes(), this->blob_bottom_->num_axes());
  EXPECT_EQ(this->blob_top_->shape(0), this->blob_bottom_->shape(0));
  EXPECT_EQ(this->blob_top_->shape(1), this->blob_bottom_->shape(1));
  EXPECT_EQ(this->blob_top_->shape(2), this->blob_bottom_->shape(2));
  EXPECT_EQ(this->blob_top_->shape(3), this->blob_bottom_->shape(3));
}


TYPED_TEST(FourierThresholdLayerTest, TestForwardZeroThreshold) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_fourier_threshold_param()->set_threshold(0);
  FourierThresholdLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(top_data[i], bottom_data[i], 1e-4);

  }
}


TYPED_TEST(FourierThresholdLayerTest, TestForwardDC1) {
  typedef typename TypeParam::Dtype Dtype;
  FillerParameter filler_param;
  filler_param.set_value(1);
  ConstantFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);

  LayerParameter layer_param;
  layer_param.mutable_fourier_threshold_param()->set_threshold(1.001);
  FourierThresholdLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], 0, 1e-4);

  }
}


TYPED_TEST(FourierThresholdLayerTest, TestForwardDC2) {
  typedef typename TypeParam::Dtype Dtype;
  FillerParameter filler_param;
  filler_param.set_value(1);
  ConstantFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);

  LayerParameter layer_param;
  layer_param.mutable_fourier_threshold_param()->set_threshold(0.99);
  FourierThresholdLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], bottom_data[i], 1e-4);

  }
}

//
//TYPED_TEST(FlattenLayerTest, TestGradient) {
//  typedef typename TypeParam::Dtype Dtype;
//  LayerParameter layer_param;
//  FlattenLayer<Dtype> layer(layer_param);
//  GradientChecker<Dtype> checker(1e-2, 1e-2);
//  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
//      this->blob_top_vec_);
//}

}  // namespace caffe
