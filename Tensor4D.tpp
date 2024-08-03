#include <iostream>
#include <vector>
#include <numeric>
#include <stdexcept>
//#include <cnpy.h>

using namespace std;

#include "Tensor4D.h"

template<typename T>
Tensor4D<T>::Tensor4D(size_t dim1, size_t dim2, size_t dim3, size_t dim4)
        : shape(Shape4D{dim1, dim2, dim3, dim4}),
          data(dim1 * dim2 * dim3 * dim4)
{}

template<typename T>
Tensor4D<T>::Tensor4D(cnpy::NpyArray data) {

}

template<typename T>
size_t Tensor4D<T>::getIndex(size_t i, size_t j, size_t k, size_t l) const {
    return ((i * shape[1] + j) * shape[2] + k) * shape[3] + l;
}

template<typename T>
T& Tensor4D<T>::operator()(size_t i, size_t j, size_t k, size_t l) {
    if (i >= shape[0] || j >= shape[1] || k >= shape[2] || l >= shape[3]) {
        throw std::out_of_range("Index out of range");
    }
    return data[getIndex(i, j, k, l)];
}

template<typename T>
const T& Tensor4D<T>::operator()(size_t i, size_t j, size_t k, size_t l) const {
    if (i >= shape[0] || j >= shape[1] || k >= shape[2] || l >= shape[3]) {
        throw std::out_of_range("Index out of range");
    }
    return data[getIndex(i, j, k, l)];
}

template<typename T>
size_t Tensor4D<T>::size(size_t dim) const {
    if (dim >= shape.size()) {
        throw std::out_of_range("Dimension out of range");
    }
    return shape[dim];
}
template<typename T>
auto Tensor4D<T>::getShape() const {
        return shape;
}

template<typename T>
void Tensor4D<T>::fill(const T& value) {
    std::fill(data.begin(), data.end(), value);
}
