#ifndef TENSOR4D_H
#define TENSOR4D_H

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cnpy.h>
#include <string>

using TensorShape = std::vector<long unsigned int>;

class Tensor4D {
private:
        std::vector<float> data_;
        TensorShape shape_;

        size_t getIndex(size_t i, size_t j, size_t k, size_t l) const;

public:
        Tensor4D();
        Tensor4D(size_t dim1, size_t dim2, size_t dim3, size_t dim4);
        Tensor4D(const cnpy::NpyArray &npy);
        Tensor4D(const Tensor4D& source);

        float& operator()(size_t i, size_t j, size_t k, size_t l);
        const float& operator()(size_t i, size_t j, size_t k, size_t l) const;

        size_t size(size_t dim) const;
        TensorShape getShape() const;
        void fill(const float& value);
        std::string toString() const;
        std::string shapeString () const;
        const float* data() const;

};

#include "Tensor4D.tpp"

#endif // TENSOR4D_H
