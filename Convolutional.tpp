
#include "Tensor4D.h"

using namespace std;

Convolutional::Convolutional(Tensor4D& weights, int stride, vector<int> padding) :
        weights_(),
        stride_(stride),
        padding_(padding),
        channels_in_(weights.getShape()[0]),
        channels_out_(weights.getShape()[1]),
        input_shape_(34, 34),
        padding_buffer_()
{
        weights_ = weights;
        padding_buffer_ = Tensor4D(1, channels_in_,
                                   input_shape_[0] + 2*padding_[0],
                                   input_shape_[1] + 2*padding_[1]);
}

Tensor4D* Convolutional::forward(const Tensor4D& input)
{
        auto input_shape = input.getShape();
        assert((input_shape[0] == input_shape_[0], "convolutional input x shape mismatch!"));
        assert((input_shape[1] == input_shape_[1], "convolutional input x shape mismatch!"));
        auto buffer = padding_buffer_.buffer();

        int q = 0;
        for (int i = 0; i < input_shape[2]; i++) {
                auto i_pad = i + padding_[0];
                for (int j = 0; j < input_shape[3]; j++) {
                        auto j_pad = j + padding_[1];
                        buffer[0, 0, i_pad, j_pad] = input(0, 0, i, j);
                        q++;
                }
        }

        cout << padding_buffer_.toString() << endl;
        return NULL;
}
