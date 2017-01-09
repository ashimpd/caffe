#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/triplet_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class TripletLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  TripletLossLayerTest()
      : blob_bottom_anchor_(new Blob<Dtype>(10, 4, 5, 2)),
        blob_bottom_positive_(new Blob<Dtype>(10, 4, 5, 2)),
        blob_bottom_negative_(new Blob<Dtype>(10, 4, 5, 2)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);

    filler.Fill(this->blob_bottom_anchor_);
    blob_bottom_vec_.push_back(blob_bottom_anchor_);

    filler.Fill(this->blob_bottom_positive_);
    blob_bottom_vec_.push_back(blob_bottom_positive_);

    filler.Fill(this->blob_bottom_negative_);
    blob_bottom_vec_.push_back(blob_bottom_negative_);

    blob_top_vec_.push_back(blob_top_loss_);
  }

  virtual ~TripletLossLayerTest() {
    delete blob_bottom_anchor_;
    delete blob_bottom_positive_;
    delete blob_bottom_negative_;
    delete blob_top_loss_;
  }

  void TestForward() {
    // Get the loss without a specified objective weight -- should be
    // equivalent to explicitly specifiying a weight of 1.
    LayerParameter layer_param;
    TripletLossLayer<Dtype> layer_weight_1(layer_param);
    layer_weight_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight_1 =
        layer_weight_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // Expect a decent loss
    EXPECT_GE(fabs(loss_weight_1), 0.1);

    // Get the loss again with a different objective weight; check that it is
    // scaled appropriately.
    const Dtype kLossWeight = 3.7;
    layer_param.add_loss_weight(kLossWeight);
    TripletLossLayer<Dtype> layer_weight_2(layer_param);
    layer_weight_2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight_2 =
        layer_weight_2.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype kErrorMargin = 1e-5;
    EXPECT_NEAR(loss_weight_1 * kLossWeight, loss_weight_2, kErrorMargin);

    // Make sure the loss is non-trivial.
    EXPECT_GE(fabs(loss_weight_1), 0.1);

    // Test with different alpha
    layer_param.mutable_threshold_param()->set_threshold(0.25);
    TripletLossLayer<Dtype> layer_weight_2a(layer_param);
    layer_weight_2a.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    Dtype loss_weight_2a =
        layer_weight_2a.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    EXPECT_GE(loss_weight_2a, loss_weight_2);
  }

  Blob<Dtype>* const blob_bottom_anchor_;
  Blob<Dtype>* const blob_bottom_positive_;
  Blob<Dtype>* const blob_bottom_negative_;
  Blob<Dtype>* const blob_top_loss_;

  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(TripletLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(TripletLossLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(TripletLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  TripletLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
