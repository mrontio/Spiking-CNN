#include "AvgPool.h"

AvgPool::AvgPool(TensorShape input_shape, shape kernel, shape stride, shape padding) :
        kernel_(kernel),
        stride_(stride),
        padding_(padding),
        timesteps_(input_shape[0]),
        input_shape_(input_shape),
        padding_buffer_(TensorShape{input_shape_[1],
                                    input_shape_[2] + 2*padding_,
                                    input_shape_[3] + 2*padding_})
{
        h_out_ = shape(floor((input_shape_[2] + 2*padding - kernel) / stride + 1));
        w_out_ = shape(floor((input_shape_[3] + 2*padding - kernel) / stride + 1));

        padding_buffer_.fill(0.0f);
}


std::unique_ptr<Tensor> AvgPool::forward(Tensor& input)
{
        auto output = std::make_unique<Tensor>(TensorShape{timesteps_, input_shape_[1], h_out_, w_out_});

        for (shape t = 0; t < timesteps_; ++t) {
                fill_padding_buffer(input(TensorShape{t}));
                for (shape c = 0; c < input_shape_[1]; ++c) {
                        for (shape i = 0; i < h_out_; ++i) {
                                for (shape j = 0; j < w_out_; ++j) {
                                        (*output)[TensorShape{t,c,i,j}] = apply_kernel(c, i, j);
                                }
                        }

                }
        }
        return output;
}


void AvgPool::fill_padding_buffer(const std::unique_ptr<Tensor> input) {
        for (shape c = 0; c < input->shape(0); ++c) {
                for (shape i = 0; i < input->shape(1); i++) {
                        shape i_pad = i + padding_;
                        for (shape j = 0; j < input->shape(2); j++) {
                                shape  j_pad = j + padding_;
                                auto input_idx = TensorShape{c, i, j};
                                auto pad_idx = TensorShape{c,i_pad,j_pad};
                                padding_buffer_[pad_idx] = (*input)[input_idx];
                        }
                }
        }

}


inline float AvgPool::apply_kernel(shape c, shape i, shape j) {
        float out = 0.0f;
        for (shape m = 0; m < kernel_; ++m) {
                for (shape n = 0; n < kernel_; ++n) {
                        out += padding_buffer_[TensorShape{c, stride_ * i + m, stride_ * j + n}];
                }
        }
        out = (1 / (float(kernel_) * float(kernel_))) * out;
        return out;
}
