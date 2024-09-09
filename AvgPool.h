#ifndef AVGPOOL_H
#define AVGPOOL_H
#include <cmath>

#include "Tensor.h"

class AvgPool {
public:
        AvgPool(TensorShape input_shape, shape kernel, shape stride, shape padding);
        std::unique_ptr<Tensor> forward(Tensor& x);

private:
        TensorShape input_shape_;
        shape kernel_;
        shape stride_;
        shape padding_;
        shape h_out_;
        shape w_out_;
        shape timesteps_;
        Tensor padding_buffer_;

        void fill_padding_buffer(const std::unique_ptr<Tensor> input);
        float apply_kernel(shape c, shape i, shape j);

};

#include "AvgPool.tpp"

#endif
