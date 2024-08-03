#ifndef TENSOR4D_H
#define TENSOR4D_H

#include <vector>
#include <stdexcept>
#include <algorithm>
//#include <cnpy.h>

using Shape4D = std::vector<long unsinged int>;
template<typename T>
class Tensor4D {
private:
        std::vector<T> data;
        Shape4D shape;

        size_t getIndex(size_t i, size_t j, size_t k, size_t l) const;

public:
        Tensor4D(size_t dim1, size_t dim2, size_t dim3, size_t dim4);
        //        Tensor4D(cnpy::NpyArray data);

        T& operator()(size_t i, size_t j, size_t k, size_t l);
        const T& operator()(size_t i, size_t j, size_t k, size_t l) const;

        size_t size(size_t dim) const;
        auto getShape() const;
        void fill(const T& value);
};

#include "Tensor4D.tpp"

#endif // TENSOR4D_H
