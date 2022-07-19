#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/flatten_layer.hpp"
#include "caffe/layers/dense_to_sparse_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template<typename TypeParam>
class DenseToSparseLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  DenseToSparseLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 4)),
        blob_bottom_2_(new Blob<Dtype>(2, 3, 6, 4)),
        blob_top_0_(new Blob<Dtype>()),
        blob_top_1_(new Blob<Dtype>()),
        blob_top_2_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(0);
    ConstantFiller<Dtype> filler(filler_param);


    filler.Fill(this->blob_bottom_);

    filler.Fill(this->blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_);

    blob_top_vec_.push_back(blob_top_0_);
    blob_top_vec_.push_back(blob_top_1_);
    blob_top_vec_.push_back(blob_top_2_);

  }

  virtual ~DenseToSparseLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_top_0_;
    delete blob_top_1_;
    delete blob_top_2_;
  }



  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_0_;
  Blob<Dtype>* const blob_top_1_;
  Blob<Dtype>* const blob_top_2_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(DenseToSparseLayerTest, TestDtypesAndDevices);

TYPED_TEST(DenseToSparseLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
    DenseToSparseParameter* dense_to_sparse_param =
      layer_param.mutable_dense_to_sparse_param();
  dense_to_sparse_param->set_fill_factor_max(.5);



  shared_ptr<Layer<Dtype> > layer(
      new DenseToSparseLayer<Dtype>(layer_param));



  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  std::vector<int> expected_shape(4);
  expected_shape[0]=2;
  expected_shape[1]=3;
  expected_shape[2]=6;
  expected_shape[3]=4;

  EXPECT_EQ(expected_shape, this->blob_top_0_->shape());
  expected_shape.resize

}

}  // namespace caffe
