#include "Convolutional.h"

using namespace std;

Convolutional::Convolutional(TensorShape input_shape, Tensor weights, shape stride, vector<shape> padding) :
        stride_(stride),
        timesteps_(input_shape[0]),
        padding_(padding),
        channels_in_(weights.shape(1)),
        channels_out_(weights.shape(0)),
        input_shape_(TensorShape{input_shape[1], input_shape[2], input_shape[3]}),
        weights_(Tensor(weights)),
        padding_buffer_(TensorShape{weights_.shape(1),
                                    input_shape_[1] + 2*padding_[0],
                                    input_shape_[2] + 2*padding_[1]}),
        kernel_size_(TensorShape{weights.shape(2), weights.shape(3)})
{

        padding_buffer_.fill(0.0f);
}

std::unique_ptr<Tensor> Convolutional::forward(const Tensor& input)
{
        // Fills padding_buffer_ with 0s and input
        // 0 is for channels_out, we keep as 0 for now
        fill_padding_buffer(input);

        shape output_w = shape((input_shape_[1] - kernel_size_[0] + 2 * padding_[0]) / stride_) + 1;
        shape output_h = shape((input_shape_[2] - kernel_size_[1] + 2 * padding_[1]) / stride_) + 1;

        auto output = std::make_unique<Tensor>(TensorShape{timesteps_, channels_out_, output_w, output_h});

        for (shape t = 0; t < timesteps_; t++) {
                for (shape l = 0; l < channels_out_; ++l) {
                        for (shape i  = 0; i < output_h; i += stride_) {
                                for (shape j  = 0; j < output_w; j += stride_) {
                                        auto output_idx = TensorShape{t,l,i,j};
                                        (*output)[output_idx] += apply_kernel(l, i, j);
                                }
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

void Convolutional::fill_padding_buffer(const Tensor& input) {
        for (shape t = 0; t < timesteps_; ++t){
                for (shape c = 0; c < channels_in_; ++c) {
                        for (shape i = 0; i < input_shape_[1]; i++) {
                                shape i_pad = i + padding_[0];
                                for (shape j = 0; j < input_shape_[2]; j++) {
                                        shape  j_pad = j + padding_[1];
                                        auto input_idx = TensorShape{t, c, i, j};
                                        auto pad_idx = TensorShape{c,i_pad,j_pad};
                                        padding_buffer_[pad_idx] = input[input_idx];
                                }
                        }
                }
        }
}
