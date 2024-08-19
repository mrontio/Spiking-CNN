#include "IntegrateFire.h"

IntegrateFire::IntegrateFire(const TensorShape input_shape) :
        input_shape_(input_shape)
{
        membrane_ = Tensor(input_shape);
}

Tensor* IntegrateFire::forward(const Tensor& x) {
        auto output = new Tensor(input_shape_);
        for (shape c = 0; c < input_shape_[0]; c++) {
                for (shape i = 0; i < input_shape_[1]; ++i) {
                        for (shape j = 0; j < input_shape_[2]; ++j) {
                                auto idx = TensorShape{c, i, j};
                                membrane_[idx] += x[idx];

                                if (membrane_[idx] > v_th_) {
                                        (*output)[idx] = 1;
                                        membrane_[idx] -= v_th_;
                                }

                                if (membrane_[idx] < min_v_mem_) {
                                        membrane_[idx] = min_v_mem_;
                                }

                        }

                }
        }
        return output;
}
