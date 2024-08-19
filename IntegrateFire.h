#ifndef INTEGRATEFIRE_H
#define INTEGRATEFIRE_H

#include "Tensor.h"

class IntegrateFire {
public:
        IntegrateFire(const TensorShape input_shape);
        Tensor* forward(const Tensor& input);

private:
        const float v_th_ = 1.0;
        const float min_v_mem_ = -1.0;
        TensorShape input_shape_;
        Tensor membrane_;
};


#include "IntegrateFire.tpp"

#endif
