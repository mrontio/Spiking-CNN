#ifndef AVGPOOL_H
#define AVGPOOL_H
#include <cmath>

#include "Tensor.h"

class AvgPool {
public:
        AvgPool(shape kernel, shape stride, shape padding);
        Tensor* forward(Tensor& x);

private:
        const TensorShape input_shape_ = {8, 34, 34};
        shape kernel_;
        shape stride_;
        shape padding_;
        shape h_out_;
        shape w_out_;
        Tensor padding_buffer_;

        void fill_padding_buffer(const Tensor& input, shape c_in);
        float apply_kernel(const Tensor& input, shape c, shape i, shape j);

};

#include "AvgPool.tpp"

#endif
