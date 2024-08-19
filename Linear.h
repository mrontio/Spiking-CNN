#ifndef LINEAR_H
#define LINEAR_H

#include "Tensor.h"

class Linear {
public:
        Linear(Tensor weights);
        Tensor* forward(const Tensor& input);

private:
        Tensor weights_;
};

#include "Linear.tpp"

#endif
