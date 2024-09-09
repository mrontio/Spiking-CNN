#include "IntegrateFire.h"


IntegrateFire::IntegrateFire(const TensorShape& input_shape) :
        input_shape_(input_shape),
        size_(Tensor::getSize(input_shape)),
        membrane_(TensorShape{Tensor::getSize(input_shape)})

{}

std::unique_ptr<Tensor> IntegrateFire::forward(Tensor& x) {
        x.flatten();
        auto output = std::make_unique<Tensor>(TensorShape{size_});
        for (shape i = 0; i < size_; ++i) {
                auto idx = TensorShape{i};
                membrane_[idx] += x[idx];

                if (membrane_[idx] >= v_th_) {
                        (*output)[idx] = 1;
                        membrane_[idx] -= v_th_;
                }

                if (membrane_[idx] < min_v_mem_) {
                        membrane_[idx] = min_v_mem_;
                }
        }

        output->reshape(input_shape_);
        return output;
}
