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
        vector<vector<float>> padding_buffer_;
        //vector<int> INPUT_SHAPE = vector<int>{34, 34};

        void fill_padding_buffer(const Tensor4D& input, int c_out, int c_in);
        float apply_kernel(int k, int l, int i, int j);

};

#include "Convolutional.tpp"

#endif
