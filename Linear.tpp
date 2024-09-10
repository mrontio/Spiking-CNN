#include "Linear.h"

Linear::Linear(Tensor weights) :
        weights_(weights)
{}

std::unique_ptr<Tensor> Linear::forward(const Tensor& input)
{
        auto timesteps = input.shape(0);
        auto output_shape = TensorShape{timesteps, weights_.shape(0)};
        auto output = make_unique<Tensor>(output_shape);
        for (shape t = 0; t < timesteps; t++) {
                for (shape c = 0; c < weights_.shape(0); ++c) {
                        float i_syn = 0.0f;
                        for (shape i = 0; i < weights_.shape(1); ++i) {
                                auto weight_idx = TensorShape{c, i};
                                i_syn += weights_[weight_idx] * input[TensorShape{t, i}];
                        }
                        (*output)[TensorShape{t, c}] = i_syn;
                }
        }
        output_shape.clear();
        return output;
}
