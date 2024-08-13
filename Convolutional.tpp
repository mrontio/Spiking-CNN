#include "Convolutional.h"

using namespace std;

Convolutional::Convolutional(Tensor weights, shape stride, vector<shape> padding) :
        stride_(stride),
        padding_(padding),
        channels_in_(weights.shape()[1]),
        channels_out_(weights.shape()[0]),
        input_shape_(34, 34)
{
        weights_ = Tensor(weights);
        TensorShape padding_shape = TensorShape{weights_.shape()[1],
                                                input_shape_[0] + 2*padding_[0],
                                                input_shape_[1] + 2*padding_[1]};
        padding_buffer_ = Tensor(padding_shape);

}

// vector<vector<vector<float>>>* Convolutional::forward(const Tensor4D& input)
// {
//         auto input_shape = input.getShape();
//         // Fills padding_buffer_
//         fill_padding_buffer(input, 0, input_shape[1]);

//         auto output = new vector<vector<vector<float>>>(channels_out_);
//         initialise_vector3d(output, channels_out_, input_shape[2], input_shape[3]);


//         for (int k = 0; k < channels_out_; ++k) {
//                 for (int l = 0; l < channels_in_; ++l) {
//                         for (int i  = 0; i < input_shape[2]; i += stride_) {
//                                 for (int j  = 0; j < input_shape[3]; j += stride_) {
//                                         (*output)[k][i][j] += apply_kernel(k, l, i, j);
//                                 }
//                         }
//                 }
//         }

//         return output;
// }

// inline float Convolutional::apply_kernel(int k, int l, int i, int j) {
//         int i_pad = i + padding_[0];
//         int j_pad = j + padding_[1];
//         auto weight_shape = weights_.getShape();
//         float output = 0.0f;
//         for (int m = 0; m < weight_shape[2]; ++m) {
//                 int m_rel = m - 1;
//                 for (int n = 0; n < weight_shape[3]; ++n) {
//                         int n_rel = n - 1;
//                         output += weights_(k, l, m, n) * padding_buffer_[l][i_pad + m_rel][j_pad + n_rel];
//                 }

//         }

//         return output;

// }

// void Convolutional::fill_padding_buffer(const Tensor4D& input, int c_out, int c_in) {
//         for (int c = 0; c < c_in; ++c) {
//                 for (int i = 0; i < input_shape_[2]; i++) {
//                         int i_pad = i + padding_[0];
//                         for (int j = 0; j < input_shape_[3]; j++) {
//                                 int  j_pad = j + padding_[1];
//                                 padding_buffer_[c][i_pad][j_pad] = input(c_out, c, i, j);
//                         }
//                 }
//         }
// }

// void Convolutional::initialise_vector3d(Vector3D* v, int a, int b, int c)
// {
//         // We assume the outer-most vector is initialised to 'a' elements
//         for (int i = 0; i < a; ++i) {
//                 (*v)[i] = vector<vector<float>>(b);
//                 for (int j = 0; j < b; ++j) {
//                         (*v)[i][j] = vector<float>(c);
//                 }
//         }
// }
