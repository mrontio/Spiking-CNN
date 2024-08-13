#ifndef CONVOLUTIONAL_H
#define CONVOLUTIONAL_H
#include <vector>
#include <iostream>
#include <assert.h>

#include "Tensor.h"

class Convolutional {
public:
        Convolutional(Tensor weights, shape stride, vector<shape> padding);
        Tensor* forward(const Tensor& input);

private:
        shape kernel_size_;
        shape stride_;
        vector<shape> padding_;
        shape channels_in_;
        shape channels_out_;
        vector<shape> input_shape_;
        Tensor weights_;
        Tensor padding_buffer_;

        void fill_padding_buffer(const Tensor& input, shape c_out, shape c_in);
        float apply_kernel(shape k, shape l, shape i, shape j);
};

#include "Convolutional.tpp"

#endif
