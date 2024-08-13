#ifndef CONVOLUTIONAL_H
#define CONVOLUTIONAL_H
#include <vector>
#include <iostream>
#include <assert.h>

#include "Tensor4D.h"

class Convolutional {
public:
        Convolutional(Tensor4D& weights, int stride, vector<int> padding);
        Tensor4D* forward(const Tensor4D& input);
private:
        int kernel_size_;
        int stride_;
        vector<int> padding_;
        int channels_in_;
        int channels_out_;
        vector<int> input_shape_;
        Tensor4D weights_;
        Tensor4D padding_buffer_;
        //vector<int> INPUT_SHAPE = vector<int>{34, 34};
};

#include "Convolutional.tpp"

#endif
