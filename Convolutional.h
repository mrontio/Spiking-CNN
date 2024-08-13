#ifndef CONVOLUTIONAL_H
#define CONVOLUTIONAL_H
#include <vector>
#include <iostream>
#include <assert.h>

#include "Tensor.h"

class Convolutional {
public:
        Convolutional(Tensor weights, shape stride, vector<shape> padding);
        // vector<vector<vector<float>>>* forward(const Tensor& input);

private:
        shape kernel_size_;
        shape stride_;
        vector<shape> padding_;
        shape channels_in_;
        shape channels_out_;
        vector<shape> input_shape_;
        Tensor weights_;
        Tensor padding_buffer_;

        // void fill_padding_buffer(const Tensor& input, int c_out, int c_in);
        // float apply_kernel(int k, int l, int i, int j);
        // void initialise_vector3d(Vector3D* v, int a, int b, int c);
};

#include "Convolutional.tpp"

#endif
