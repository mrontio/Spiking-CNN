#ifndef CONVOLUTIONAL_H
#define CONVOLUTIONAL_H
#include <vector>
#include <iostream>
#include <assert.h>

#include "Tensor.h"

class Convolutional {
public:
        Convolutional(TensorShape input_shape, Tensor weights, shape stride, vector<shape> padding);
        std::unique_ptr<Tensor> forward(const Tensor& input);

private:
        TensorShape kernel_size_;
        shape stride_;
        TensorShape padding_;
        shape timesteps_;
        shape channels_in_;
        shape channels_out_;
        TensorShape input_shape_;
        Tensor weights_;
        Tensor padding_buffer_;


        void fill_padding_buffer(const Tensor& input);

        float apply_kernel(shape l, shape i, shape j);
};

#include "Convolutional.tpp"

#endif
