#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cnpy.h>
#include <string>

using TensorShape = std::vector<long unsigned int>;

class Tensor {
private:
        TensorShape shape_;
        unsigned int N_;
        std::vector<float> data_;


        size_t getIndex(TensorShape dims) const;

public:
        Tensor();
        Tensor(TensorShape dims);
        Tensor(const cnpy::NpyArray &npy);
        Tensor(const Tensor& source);

        float& operator[](TensorShape dims);
        const float operator[](TensorShape dims) const;

        TensorShape shape() const;
        std::string shapeString () const;
        void fill(const float& value);


        const float* data() const;

};

#include "Tensor.tpp"

#endif // TENSOR_H
