#include <iostream>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <cnpy.h>

using namespace std;

#include "Tensor4D.h"


Tensor4D::Tensor4D()
        : shape_(TensorShape{0,0,0,0}),
          data_()
{}

Tensor4D::Tensor4D(size_t dim1, size_t dim2, size_t dim3, size_t dim4)
        : shape_(TensorShape{dim1, dim2, dim3, dim4}),
          data_(dim1 * dim2 * dim3 * dim4)
{}

Tensor4D::Tensor4D(const cnpy::NpyArray& npy)
        : shape_(npy.shape)
{
        if (npy.shape.size() != shape_.size()) {
                throw std::out_of_range("NpyArray is not 4-dimensional");
        }

        const float* elements = npy.data<float>();
        data_ = std::vector<float>(elements, elements + (shape_[0] * shape_[1] * shape_[2] * shape_[3]));
}

Tensor4D::Tensor4D(const Tensor4D& source)
        : shape_(source.getShape())
{
        const float* data = source.data();
        data_= std::vector<float>(data, data+ (shape_[0] * shape_[1] * shape_[2] * shape_[3]));
}

size_t Tensor4D::getIndex(size_t i, size_t j, size_t k, size_t l) const {
    return ((i * shape_[1] + j) * shape_[2] + k) * shape_[3] + l;
}

float& Tensor4D::operator()(size_t i, size_t j, size_t k, size_t l) {
    if (i >= shape_[0] || j >= shape_[1] || k >= shape_[2] || l >= shape_[3]) {
        throw std::out_of_range("Index out of range");
    }
    return data_[getIndex(i, j, k, l)];
}

const float& Tensor4D::operator()(size_t i, size_t j, size_t k, size_t l) const {
    if (i >= shape_[0] || j >= shape_[1] || k >= shape_[2] || l >= shape_[3]) {
        throw std::out_of_range("Index out of range");
    }
    return data_[getIndex(i, j, k, l)];
}

size_t Tensor4D::size(size_t dim) const {
    if (dim >= shape_.size()) {
        throw std::out_of_range("Dimension out of range");
    }
    return shape_[dim];
}

TensorShape Tensor4D::getShape() const {
        return shape_;
}

float const * Tensor4D::data() const {
        return data_.data();
}

const std::vector<float>& Tensor4D::buffer() {
        return data_;
}

void Tensor4D::fill(const float& value) {
    std::fill(data_.begin(), data_.end(), value);
}

std::string Tensor4D::toString() const {
    std::ostringstream os;
    os << "Tensor4D(";
    for (size_t i = 0; i < shape_[0]; ++i) {
        os << "[";
        for (size_t j = 0; j < shape_[1]; ++j) {
            os << "[";
            for (size_t k = 0; k < shape_[2]; ++k) {
                os << "[";
                for (size_t l = 0; l < shape_[3]; ++l) {
                    os << (*this)(i, j, k, l);
                    if (l < shape_[3] - 1) os << ", ";
                }
                os << "]";
                if (k < shape_[2] - 1) os << ", ";
            }
            os << "]";
            if (j < shape_[1] - 1) os << ", ";
        }
        os << "]";
        if (i < shape_[0] - 1) os << ", ";
    }
    os << ")";
    return os.str();
}

std::string Tensor4D::shapeString() const {
        std::ostringstream os;
        os << "("
           << shape_[0] << ","
           << shape_[1] << ","
           << shape_[2] << ","
           << shape_[3]
           << ")";
        return os.str();
}
