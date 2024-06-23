/*
Permutation layer to be used for the interpretation of the blobs extracted
from ConvDet, which are extracted from the last Convolutional Layer, and have
to be splitted before hand. This layer is with the purpose of actually
re-arranging the data as in tensorflow format, which is the one used for
SqueezeDet, otherwise the results wont match and will get incorrent and
a lot of false alarms.

Author: weiliu89
modified: Josue Cuevas
ConvDet oriented rather than multibox SSD
*/
#ifndef CAFFE_PERMUTATION_LAYER_HPP_
#define CAFFE_PERMUTATION_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Permute the input blob by changing the memory order of the data.
 * Remember: ConvDet inside SqueezeDet was arranged as [Batch, Height, Width, Channels]
 * but caffe memory mapping is [Batch, Channels, Height, Width]
 * this should facilitate the access and correct replication of the predictions
 * when extracting the answer from the last convolutional layer of ConvDet
 */

  /*
  Main permutation function to be called when layer specified inside the
  prototxt "Call this function as Permute"
  */
  template <typename Dtype>
  void Permute(const int count, Dtype* bottom_data,
    const bool forward, const int* permute_order, const int* old_steps,
    const int* new_steps, const int num_axes, Dtype* top_data);

  /*
  Functions to be access during implementatio of this layer. Accessible by
  the *.cpp and *.cu files
  */
  template <typename Dtype>
  class PermuteLayer : public Layer<Dtype>{
    public:
      explicit PermuteLayer(const LayerParameter& param): Layer<Dtype>(param){}
      virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
      virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

      /*
       Layer declaration, and required blobs (top/bottom) for
       the layer to be called without problem
      */
      virtual inline const char* type() const { return "Permute"; }
      virtual inline int ExactNumBottomBlobs() const { return 1; }
      virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
      /*
      Forward pass for CPU implementation, implementation is to be done
      somewhere else
      */
      virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
        /*
        Forward pass for GPU implementation, implementation is to be done
        somewhere else
        */
      virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

      /*
      Backward propagation for CPU implementation, implementation is to be done
      somewhere else
      */
      virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

      /*
      Backward propagation for GPU implementation, implementation is to be done
      somewhere else
      */
      virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    int num_axes_;// number of axes to be used during permutation
    bool need_permute_;// Flag to make sure permutation is done

    // Use Blob because it is convenient to be accessible in .cu file.
    Blob<int> permute_order_;
    Blob<int> old_steps_;
    Blob<int> new_steps_;
  };

}  // namespace caffe

#endif  // CAFFE_PERMUTATION_LAYER_HPP_
