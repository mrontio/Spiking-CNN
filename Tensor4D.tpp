#include <iostream>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <cnpy.h>

using namespace std;

#include "Tensor4D.h"

Tensor4D::Tensor4D(size_t dim1, size_t dim2, size_t dim3, size_t dim4)
        : shape(TensorShape{dim1, dim2, dim3, dim4}),
          data(dim1 * dim2 * dim3 * dim4)
{}

Tensor4D::Tensor4D(const cnpy::NpyArray& npy)
        : shape(npy.shape)
{
        if (npy.shape.size() != shape.size()) {
                throw std::out_of_range("NpyArray is not 4-dimensional");
        }

        const float* elements = npy.data<float>();
        data = std::vector<float>(elements, elements + (shape[0] * shape[1] * shape[2] * shape[3]));
}

size_t Tensor4D::getIndex(size_t i, size_t j, size_t k, size_t l) const {
    return ((i * shape[1] + j) * shape[2] + k) * shape[3] + l;
}

float& Tensor4D::operator()(size_t i, size_t j, size_t k, size_t l) {
    if (i >= shape[0] || j >= shape[1] || k >= shape[2] || l >= shape[3]) {
        throw std::out_of_range("Index out of range");
    }
    return data[getIndex(i, j, k, l)];
}

const float& Tensor4D::operator()(size_t i, size_t j, size_t k, size_t l) const {
    if (i >= shape[0] || j >= shape[1] || k >= shape[2] || l >= shape[3]) {
        throw std::out_of_range("Index out of range");
    }
    return data[getIndex(i, j, k, l)];
}

size_t Tensor4D::size(size_t dim) const {
    if (dim >= shape.size()) {
        throw std::out_of_range("Dimension out of range");
    }
    return shape[dim];
}

auto Tensor4D::getShape() const {
        return shape;
}

void Tensor4D::fill(const float& value) {
    std::fill(data.begin(), data.end(), value);
}

std::string Tensor4D::toString() const {
    std::ostringstream os;
    os << "Tensor4D(";
    for (size_t i = 0; i < shape[0]; ++i) {
        os << "[";
        for (size_t j = 0; j < shape[1]; ++j) {
            os << "[";
            for (size_t k = 0; k < shape[2]; ++k) {
                os << "[";
                for (size_t l = 0; l < shape[3]; ++l) {
                    os << (*this)(i, j, k, l);
                    if (l < shape[3] - 1) os << ", ";
                }
                os << "]";
                if (k < shape[2] - 1) os << ", ";
            }
            os << "]";
            if (j < shape[1] - 1) os << ", ";
        }
        os << "]";
        if (i < shape[0] - 1) os << ", ";
    }
    os << ")";
    return os.str();
}

std::string Tensor4D::shapeString() const {
        std::ostringstream os;
        os << "("
           << shape[0] << ","
           << shape[1] << ","
           << shape[2] << ","
           << shape[3]
           << ")";
        return os.str();
}
