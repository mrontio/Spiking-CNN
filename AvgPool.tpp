#include "AvgPool.h"

AvgPool::AvgPool(TensorShape input_shape, shape kernel, shape stride, shape padding) :
        kernel_(kernel),
        stride_(stride),
        padding_(padding),
        input_shape_(input_shape)
{
        h_out_ = shape(floor((input_shape_[1] + 2*padding - kernel) / stride + 1));
        w_out_ = shape(floor((input_shape_[2] + 2*padding - kernel) / stride + 1));

        TensorShape padding_shape = TensorShape{input_shape_[0],
                                                input_shape_[1] + 2*padding_,
                                                input_shape_[2] + 2*padding_};
        padding_buffer_ = Tensor(padding_shape);
        padding_buffer_.fill(0.0f);
}


Tensor* AvgPool::forward(Tensor& input)
{
        fill_padding_buffer(input, input_shape_[0]);
        auto output = new Tensor(TensorShape{input_shape_[0], h_out_, w_out_});

        for (shape c = 0; c < input_shape_[0]; ++c) {
                for (shape i = 0; i < h_out_; ++i) {
                        for (shape j = 0; j < w_out_; ++j) {
                                (*output)[TensorShape{c,i,j}] = apply_kernel(input, c, i, j);
                        }
                }

        }
        return output;
}


void AvgPool::fill_padding_buffer(const Tensor& input, shape c_in) {
        for (shape c = 0; c < input_shape_[0]; ++c) {
                for (shape i = 0; i < input_shape_[1]; i++) {
                        shape i_pad = i + padding_;
                        for (shape j = 0; j < input_shape_[2]; j++) {
                                shape  j_pad = j + padding_;
                                auto input_idx = TensorShape{c, i, j};
                                auto pad_idx = TensorShape{c,i_pad,j_pad};
                                padding_buffer_[pad_idx] = input[input_idx];
                        }
                }
        }

}


inline float AvgPool::apply_kernel(const Tensor& input, shape c, shape i, shape j) {
        float out = 0.0f;
        for (shape m = 0; m < kernel_; ++m) {
                for (shape n = 0; n < kernel_; ++n) {
                        out += input[TensorShape{c, stride_ * i + m, stride_ * j + n}];
                }
        }
        out = (1 / (float(kernel_) * float(kernel_))) * out;
        return out;
}
