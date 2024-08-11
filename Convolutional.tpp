
#include "Tensor4D.h"

using namespace std;

Convolutional::Convolutional(Tensor4D& weights, int stride, vector<int> padding)
        : weights_(weights), stride_(stride), padding_(padding)
{
        auto w_shape = weights_.getShape();
        input_shape_ = INPUT_SHAPE;
        channels_in = w_shape[1];
        channels_in = w_shape[0];
        int y_pad = input_shape_[0] + 2*padding_[0];
        int x_pad = input_shape_[1] + 2*padding_[1];
        padding_buffer_ = vector<vector<float>>(y_pad, vector<float>(y_pad, 0.0f));
}

Tensor4D* Convolutional::forward(const Tensor4D& input)
{
        auto input_shape = input.getShape();
        assert((input_shape[0] == input_shape_[0], "convolutional input x shape mismatch!"));
        assert((input_shape[1] == input_shape_[1], "convolutional input x shape mismatch!"));

        //for (int i = 0; i < get<0>()[0]

        return NULL;
}
