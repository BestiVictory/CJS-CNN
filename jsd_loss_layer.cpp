#include <vector>

#include "caffe/layers/jsd_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void JSDLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void JSDLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  
  Dtype loss = 0;
    
  for (int i = 0; i < count; ++i) {
    if(input_data[i] <= 0.00001){
        loss += target[i] * log(2.0);
        continue;
    }    
    if(target[i] <= 0.00001){
        loss += input_data[i] * log(2.0);
        continue;
    }    
    loss += (input_data[i] * log(2*input_data[i]) + target[i] * log(2*target[i]) - (input_data[i] + target[i]) * log(input_data[i] + target[i]));
  }
  top[0]->mutable_cpu_data()[0] = loss / 2 / num;
}

template <typename Dtype>
void JSDLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    
    const Dtype* input_data = bottom[0]->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	
    for (int i = 0; i < count; ++i) { 
            if(target[i] <= 0.00001){
                bottom_diff[i] = 0.5 * log(2.0);
                continue;
            }    
            /*if(input_data[i] <= 0.00001){
                bottom_diff[i] = -log(1-target[i]);
                continue;
            } */   
            bottom_diff[i] = 0.5 * log(2*input_data[i]/(input_data[i]+target[i]));
    }
    
    const Dtype loss_weight = top[0]->cpu_diff()[0];

    for(int i=0;i<count;++i){
        int dis = 9 - i%10;
        bottom_diff[i] = bottom_diff[i] * loss_weight / num;
    }          
    
    /* 
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight / num, bottom_diff);
    */
  }
}


INSTANTIATE_CLASS(JSDLossLayer);
REGISTER_LAYER_CLASS(JSDLoss);

}  // namespace caffe
