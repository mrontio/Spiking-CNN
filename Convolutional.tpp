
#include "Tensor4D.h"

using namespace std;

Convolutional::Convolutional(Tensor4D& weights, int stride, vector<int> padding) :
        weights_(),
        stride_(stride),
        padding_(padding),
        channels_in_(weights.getShape()[1]),
        channels_out_(weights.getShape()[0]),
        input_shape_(34, 34),
        padding_buffer_()
{
        weights_ = weights;
        padding_buffer_.resize(input_shape_[0] + 2*padding_[0]);
        for (int i = 0; i < padding_buffer_.size(); i++) {
                padding_buffer_[i].resize(input_shape_[1] + 2*padding_[1]);
        }
}

vector<vector<vector<float>>>* Convolutional::forward(const Tensor4D& input)
{
        auto input_shape = input.getShape();
        // Fills padding_buffer_
        fill_padding_buffer(input, 0, 0);
        //


        // TODO: loop over channels as well.
        // For the moment we're just doing negative polarity (AER) and first output channel
        // You need to change the padding buffer as well
        auto output = new vector<vector<vector<float>>>(channels_in_);
        for (int c = 0; c < channels_in_; ++c) {
                (*output)[c] = vector<vector<float>>(input_shape[2]);
                for (int i = 0; i < input_shape[2]; ++i) {
                        (*output)[c][i] = vector<float>(input_shape[3]);
                }
        }

        int k = 0;
        int l = 0;
        for (int i  = 0; i < input_shape[2]; i += stride_) {
                for (int j  = 0; j < input_shape[3]; j += stride_) {
                        (*output)[k][i][j] = apply_kernel(k, l, i, j);
                }
        }

        return output;
}

inline float Convolutional::apply_kernel(int k, int l, int i, int j) {
        int i_pad = i + padding_[0];
        int j_pad = j + padding_[1];
        auto weight_shape = weights_.getShape();
        float output = 0.0f;
        for (int m = 0; m < weight_shape[2]; ++m) {
                int m_rel = m - 1;
                for (int n = 0; n < weight_shape[3]; ++n) {
                        int n_rel = n - 1;
                        output += weights_(k, l, m, n) * padding_buffer_[i_pad + m_rel][j_pad + n_rel];
                }

        }

        return output;

}

void Convolutional::fill_padding_buffer(const Tensor4D& input, int c_out, int c_in) {
        for (int c = 0; c < c_in; ++c) {
                for (int i = 0; i < input_shape_[2]; i++) {
                        auto i_pad = i + padding_[0];
                        for (int j = 0; j < input_shape_[3]; j++) {
                                auto j_pad = j + padding_[1];
                                padding_buffer_[c][i_pad][j_pad] = input(c_out, c_in, i, j);
                        }
                }
        }
}
