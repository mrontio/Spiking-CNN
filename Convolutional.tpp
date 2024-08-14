#include "Convolutional.h"

using namespace std;

Convolutional::Convolutional(Tensor weights, shape stride, vector<shape> padding) :
        stride_(stride),
        padding_(padding),
        channels_in_(weights.shape(1)),
        channels_out_(weights.shape(0)),
        input_shape_(34, 34)
{
        weights_ = Tensor(weights);
        TensorShape padding_shape = TensorShape{weights_.shape(1),
                                                input_shape_[0] + 2*padding_[0],
                                                input_shape_[1] + 2*padding_[1]};
        padding_buffer_ = Tensor(padding_shape);
        padding_buffer_.fill(0.0f);

}

Tensor* Convolutional::forward(const Tensor& input)
{
        auto input_shape = input.shape();
        // Fills padding_buffer_ with 0s and input
        // 0 is for channels_out, we keep as 0 for now
        fill_padding_buffer(input, 0, channels_in_);

        auto output = new Tensor(TensorShape{channels_out_, input_shape[2], input_shape[3]});

        for (shape l = 0; l < channels_out_; ++l) {
                for (shape i  = 0; i < input_shape[2]; i += stride_) {
                        for (shape j  = 0; j < input_shape[3]; j += stride_) {
                                auto output_idx = TensorShape{l,i,j};
                                (*output)[output_idx] += apply_kernel(l, i, j);
                        }
                }
        }

        return output;
}

inline float Convolutional::apply_kernel(shape l, shape i, shape j) {
        shape i_pad = i + padding_[0];
        shape j_pad = j + padding_[1];
        float output = 0.0f;
        for (shape k = 0; k < channels_in_; ++k) {
                for (shape m = 0; m < weights_.shape(2); ++m) {
                        shape m_rel = m - 1;
                        for (shape n = 0; n < weights_.shape(3); ++n) {
                                shape n_rel = n - 1;
                                auto weight_idx = TensorShape{l,k,m,n};
                                auto pad_idx = TensorShape{k, i_pad + m_rel, j_pad + n_rel};
                                auto weight = weights_[weight_idx];
                                auto value = padding_buffer_[pad_idx];
                                output += weight * value ;
                        }
                }
        }
        return output;
}

void Convolutional::fill_padding_buffer(const Tensor& input, shape c_out, shape c_in) {
        for (shape c = 0; c < c_in; ++c) {
                for (shape i = 0; i < input_shape_[2]; i++) {
                        shape i_pad = i + padding_[0];
                        for (shape j = 0; j < input_shape_[3]; j++) {
                                shape  j_pad = j + padding_[1];
                                auto input_idx = TensorShape{c_out, c, i, j};
                                auto pad_idx = TensorShape{c,i_pad,j_pad};
                                padding_buffer_[pad_idx] = input[input_idx];
                        }
                }
        }
}
