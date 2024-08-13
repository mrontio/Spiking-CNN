#ifndef CONVOLUTIONAL_H
#define CONVOLUTIONAL_H
#include <vector>
#include <iostream>
#include <assert.h>

#include "Tensor4D.h"

using Vector3D = vector<vector<vector<float>>>;

class Convolutional {
public:
        Convolutional(Tensor4D& weights, int stride, vector<int> padding);
        vector<vector<vector<float>>>* forward(const Tensor4D& input);

private:
        int kernel_size_;
        int stride_;
        vector<int> padding_;
        int channels_in_;
        int channels_out_;
        vector<int> input_shape_;
        Tensor4D weights_;
        Vector3D padding_buffer_;

        void fill_padding_buffer(const Tensor4D& input, int c_out, int c_in);
        float apply_kernel(int k, int l, int i, int j);
        void initialise_vector3d(Vector3D* v, int a, int b, int c);
};

#include "Convolutional.tpp"

#endif
