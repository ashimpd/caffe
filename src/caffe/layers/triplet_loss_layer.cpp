#include <vector>

#include "caffe/layers/triplet_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TripletLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  alpha_ = this->layer_param_.threshold_param().threshold();
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK(bottom[0]->shape() == bottom[1]->shape())
      << "Inputs 0 and 1 dont have same shape.";
  CHECK(bottom[1]->shape() == bottom[2]->shape())
      << "Inputs 1 and 2 dont have same shape.";
  positive_class_diff_.ReshapeLike(*bottom[0]);
  negative_class_diff_.ReshapeLike(*bottom[0]);

  vector<int> top_shape(0); // Single value loss
  top[0]->Reshape(top_shape);

  batch_size_ = bottom[0]->shape(0);
  vec_size_ = bottom[0]->count() / batch_size_;
  loss_vec_.resize(batch_size_);
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  int count = bottom[0]->count();

  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      positive_class_diff_.mutable_cpu_data());

  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[2]->cpu_data(),
      negative_class_diff_.mutable_cpu_data());

  Dtype loss = 0;
  for(int i = 0; i < batch_size_; i++) {
    Dtype d1 = caffe_cpu_dot(vec_size_,
                             positive_class_diff_.cpu_data() + i * vec_size_,
                             positive_class_diff_.cpu_data() + i * vec_size_);

    Dtype d2 = caffe_cpu_dot(vec_size_,
                             negative_class_diff_.cpu_data() + i * vec_size_,
                             negative_class_diff_.cpu_data() + i * vec_size_);

    loss_vec_[i] = alpha_ + d1 - d2;
    if(loss_vec_[i] < 0)
        loss_vec_[i] = 0;
    loss += loss_vec_[i];
  }

  loss = loss / batch_size_ / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  Dtype scale = top[0]->cpu_diff()[0] / bottom[0]->num();
  int count = bottom[0]->count();

  caffe_sub(count, positive_class_diff_.cpu_data(),
            negative_class_diff_.cpu_data(),
            bottom[0]->mutable_cpu_diff());

  caffe_scal(count, scale, bottom[0]->mutable_cpu_diff());

  caffe_cpu_scale(count, -scale, positive_class_diff_.cpu_data(),
                  bottom[1]->mutable_cpu_diff());
  
  caffe_cpu_scale(count, scale, negative_class_diff_.cpu_data(),
                  bottom[2]->mutable_cpu_diff());

  for(int i = 0; i < batch_size_; i++) {
    if(loss_vec_[i] == 0) {
      caffe_set(vec_size_, Dtype(0), bottom[0]->mutable_cpu_diff() + i*vec_size_);
      caffe_set(vec_size_, Dtype(0), bottom[1]->mutable_cpu_diff() + i*vec_size_);
      caffe_set(vec_size_, Dtype(0), bottom[2]->mutable_cpu_diff() + i*vec_size_);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(TripletLossLayer);
#endif

INSTANTIATE_CLASS(TripletLossLayer);
REGISTER_LAYER_CLASS(TripletLoss);

}  // namespace caffe
