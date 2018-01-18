/*
  CPU implementation for the layer permutation in caffe, to match the memory
  mapping from ConvDet used in SqueezeDet. The purpose of this is to be used
  after splitting the access, and rearrange the memory for matching that of
  tensorflow, in order to extract the Boxes, classes-probs, and confidences
  which are later input in the NMS part of the inference script

  Author: weiliu89
  modified: Josue Cuevas
  ConvDet oriented rather than multibox SSD
*/
#include <vector>
#include "caffe/layers/permute_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
  /*
  Main permutation for cpu implementation deciding if calling backward or
  forward passes.
  */
  template <typename Dtype>
  void Permute(const int count, Dtype* bottom_data, const bool forward,
    const int* permute_order, const int* old_steps, const int* new_steps,
    const int num_axes, Dtype* top_data){
      for(int i = 0; i < count; ++i){
        int old_idx = 0;
        int idx = i;
        for (int j = 0; j < num_axes; ++j){
          int order = permute_order[j];//using the order specified by user

          /*
          data indexing estimation for new data mapping
          */
          old_idx += (idx / new_steps[j]) * old_steps[order];
          idx %= new_steps[j];
        }
        if(forward){//invoke forward pass
          top_data[i] = bottom_data[old_idx];
        }else{//invoke backward prop
          bottom_data[old_idx] = top_data[i];
        }
      }
  }

  /*
  Setting up layer once for correct execution and results
  */
  template <typename Dtype>
  void PermuteLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
      //parameter defined in caffe.proto line ~391
      //"optional PermuteParameter permute_param = 202;"
      PermuteParameter permute_param = this->layer_param_.permute_param();

      // make sure no multiblob is input inside this layer
      CHECK_EQ(bottom.size(), 1);

      // axis in this blob according to caffe coding standard
      num_axes_ = bottom[0]->num_axes();
      // all orders or data arrangement indexes are to be saved here
      vector<int> orders;

      // Push the specified new orders.
      for(int i = 0; i < permute_param.order_size(); ++i){
        // order to be used, should be no more than the number of axes
        int order = permute_param.order(i);

        // make sure user don't input more parmutations than axes
        CHECK_LT(order, num_axes_)
          << "order should be less than the input dimension.";

        // Make sure user doesn't layer twice in the order
        if(std::find(orders.begin(), orders.end(), order) != orders.end()){
          LOG(FATAL) << "there are duplicate orders";
        }
        orders.push_back(order);
      }

      // Push the rest orders. And save original step sizes for each axis.
      for(int i = 0; i < num_axes_; ++i){
        if(std::find(orders.begin(), orders.end(), i) == orders.end()){
          // queuing the order in which the data is gonna be arranged
          orders.push_back(i);
        }
      }
      CHECK_EQ(num_axes_, orders.size());

      // Check if we need to reorder the data or keep it.
      need_permute_ = false;
      for(int i = 0; i < num_axes_; ++i){
        if(orders[i] != i){
          // As long as there is one order which is different from the natural order
          // of the data, we need to permute. Otherwise, we share the data and diff.
          need_permute_ = true;
          break;
        }
      }

      // reshaping data to required order
      vector<int> top_shape(num_axes_, 1);
      permute_order_.Reshape(num_axes_, 1, 1, 1);
      old_steps_.Reshape(num_axes_, 1, 1, 1);
      new_steps_.Reshape(num_axes_, 1, 1, 1);
      for (int i = 0; i < num_axes_; ++i) {
        // getting data indexing and data shapes
        permute_order_.mutable_cpu_data()[i] = orders[i];
        top_shape[i] = bottom[0]->shape(orders[i]);
      }
      top[0]->Reshape(top_shape);
  }

  /*
  To be used during setting up and permutation implementation
  */
  template <typename Dtype>
  void PermuteLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
      vector<int> top_shape;
      // shapes for outgoing data from the current arrangement
      for(int i = 0; i < num_axes_; ++i){
        if(i == num_axes_ - 1){
          old_steps_.mutable_cpu_data()[i] = 1;
        }else{
          old_steps_.mutable_cpu_data()[i] = bottom[0]->count(i + 1);
        }
        top_shape.push_back(bottom[0]->shape(permute_order_.cpu_data()[i]));
      }

      // actual reshape of the data to be used in the layer
      top[0]->Reshape(top_shape);

      // shapes for outgoing data from the current arrangement
      for(int i = 0; i < num_axes_; ++i){
        if(i == num_axes_ - 1){
          new_steps_.mutable_cpu_data()[i] = 1;
        }else{
          new_steps_.mutable_cpu_data()[i] = top[0]->count(i + 1);
        }
      }
  }

  /*
  Forward pass implementation for this layer, CPU based. When the layer
  is done with a particular axes or the axis doesn't need permutation the data
  is directly shared
  */
  template <typename Dtype>
  void PermuteLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    if(need_permute_){
      /*
      Actual permutation of any layer which is not in the predefined
      caffe order. accesing data
      */
      Dtype* bottom_data = bottom[0]->mutable_cpu_data();
      Dtype* top_data = top[0]->mutable_cpu_data();
      const int top_count = top[0]->count();
      const int* permute_order = permute_order_.cpu_data();
      const int* old_steps = old_steps_.cpu_data();
      const int* new_steps = new_steps_.cpu_data();
      bool forward = true;

      // calling main function to change or switch data
      Permute(top_count, bottom_data, forward, permute_order, old_steps,
        new_steps, num_axes_, top_data);
    }else{
      // If there is no need to permute, we share data to save memory.
      top[0]->ShareData(*bottom[0]);
    }
  }

  /*
  Backward implementation of the permutation layer, where all the errors are
  to be passed to the blobs in the order before permutation, otherwise we will
  add/subs in the wrong location of the memory mapped
  */
  template <typename Dtype>
  void PermuteLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
      if(need_permute_){
        // Accessing gradients
        Dtype* top_diff = top[0]->mutable_cpu_diff();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        const int top_count = top[0]->count();
        const int* permute_order = permute_order_.cpu_data();
        const int* old_steps = old_steps_.cpu_data();
        const int* new_steps = new_steps_.cpu_data();
        bool forward = false;
        Permute(top_count, bottom_diff, forward, permute_order, old_steps,
                new_steps, num_axes_, top_diff);
      }else{
        // If there is no need to permute, we share diff to save memory.
        bottom[0]->ShareDiff(*top[0]);
    }
  }

  #ifdef CPU_ONLY
    STUB_GPU(PermuteLayer);
  #endif

  // Instantiation and registration of this layer
  INSTANTIATE_CLASS(PermuteLayer);
  REGISTER_LAYER_CLASS(Permute);

}  // namespace caffe
