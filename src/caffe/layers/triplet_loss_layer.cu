#include <vector>

#include "caffe/layers/triplet_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TripletLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  int count = bottom[0]->count();

  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      positive_class_diff_.mutable_gpu_data());

  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[2]->gpu_data(),
      negative_class_diff_.mutable_gpu_data());

  Dtype loss = 0;
  for(int i = 0; i < batch_size_; i++) {
    Dtype d1;
    caffe_gpu_dot(vec_size_,
                  positive_class_diff_.gpu_data() + i * vec_size_,
                  positive_class_diff_.gpu_data() + i * vec_size_);

    Dtype d2;
    caffe_gpu_dot(vec_size_,
                  negative_class_diff_.gpu_data() + i * vec_size_,
                  negative_class_diff_.gpu_data() + i * vec_size_);

    loss_vec_[i] = alpha_ + d1 - d2;
    if(loss_vec_[i] < 0)
        loss_vec_[i] = 0;
    loss += loss_vec_[i];
  }

  Dtype loss = loss / batch_size_ / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  Dtype scale = top[0]->cpu_diff()[0] / bottom[0]->num();
  int count = bottom[0]->count();

  caffe_gpu_sub(count, positive_class_diff_.gpu_data(),
                negative_class_diff_.gpu_data(),
                bottom[0]->mutable_gpu_diff());

  caffe_gpu_scal(count, scale, bottom[0]->mutable_gpu_diff());

  caffe_gpu_scale(count, -scale, positive_class_diff_.gpu_data(),
                  bottom[1]->mutable_gpu_diff());
  
  caffe_gpu_scale(count, scale, negative_class_diff_.gpu_data(),
                  bottom[2]->mutable_gpu_diff());

  for(int i = 0; i < batch_size_; i++) {
    if(loss_vec_[i] == 0) {
      caffe_gpu_set(vec_size_, Dtype(0), bottom[0]->mutable_gpu_diff() + i*vec_size_);
      caffe_gpu_set(vec_size_, Dtype(0), bottom[1]->mutable_gpu_diff() + i*vec_size_);
      caffe_gpu_set(vec_size_, Dtype(0), bottom[2]->mutable_gpu_diff() + i*vec_size_);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(TripletLossLayer);

}  // namespace caffe
