#include "Linear.h"

Linear::Linear(Tensor weights) :
        weights_(weights)
{}

Tensor* Linear::forward(const Tensor& input)
{
        auto output = new Tensor(TensorShape{weights_.shape(0)});
        for (shape c = 0; c < weights_.shape(0); ++c) {
                float i_syn = 0.0f;
                for (shape i = 0; i < weights_.shape(1); ++i) {
                        auto weight_idx = TensorShape{c, i};
                        i_syn += weights_[weight_idx] * input[TensorShape{i}];
                }
                (*output)[TensorShape{c}] = i_syn;
        }
        return output;
}
