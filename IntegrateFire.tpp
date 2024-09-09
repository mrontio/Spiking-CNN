#include "IntegrateFire.h"


IntegrateFire::IntegrateFire(const TensorShape& input_shape) :
        input_shape_(input_shape),
        timesteps_(input_shape[0]),
        data_shape_(Tensor::shave(input_shape, 1)),
        size_(Tensor::getSize(data_shape_)),
        membrane_(TensorShape{Tensor::getSize(Tensor::shave(input_shape, 1))})
{
}

std::unique_ptr<Tensor> IntegrateFire::forward(Tensor& x) {
        TensorShape s = TensorShape{timesteps_, Tensor::getSize(data_shape_)};
        x.reshape(s);
        auto output = std::make_unique<Tensor>(s);
        for (shape t = 0; t < timesteps_; ++t) {
                for (shape i = 0; i < size_; ++i) {
                        auto mem_idx = TensorShape{i};
                        auto data_idx = TensorShape{t, i};
                        membrane_[mem_idx] += x[data_idx];

                        if (membrane_[mem_idx] >= v_th_) {
                                (*output)[data_idx] = 1;
                                membrane_[mem_idx] -= v_th_;
                        }

                        if (membrane_[mem_idx] < min_v_mem_) {
                                membrane_[mem_idx] = min_v_mem_;
                        }
                }
        }

        output->reshape(input_shape_);
        return output;
}
