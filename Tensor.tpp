#include <iostream>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <cnpy.h>
#include <cmath>
#include <format>

using namespace std;

#include "Tensor.h"


Tensor::Tensor()
        : shape_(TensorShape()),
          data_()
{}

Tensor::Tensor(const TensorShape dims)
        : shape_(dims)
{
        int size = 1;
        for (int n = 0; n < shape_.size(); n++) {
                size *= dims[n];
        }
        data_ = std::vector<float>(size);
}


Tensor::Tensor(const cnpy::NpyArray& npy)
        : shape_(npy.shape)
{
        auto size = getSize(shape_);

        const float* elements = npy.data<float>();
        data_ = std::vector<float>(elements, elements + size );
}

Tensor::Tensor(const Tensor& source)
        : shape_(source.shape())
{
        const float* data = source.data();
        data_= std::vector<float>(data, data+ (shape_[0] * shape_[1] * shape_[2] * shape_[3]));
}


Tensor& Tensor::reshape(const TensorShape shape) {
        auto size = getSize(shape);
        if (size != data_.size()) {
                throw std::runtime_error("shape : element mismatch");
        }
        shape_ = shape;
        return *this;
}

Tensor& Tensor::flatten() {
        shape_ = TensorShape{getSize(shape_)};
        return *this;
}

size_t Tensor::getIndex(TensorShape dims) const
{
        int v_index = dims[0];
        for (int i = 1; i < shape_.size(); ++i) {
                v_index = v_index * shape_[i] + dims[i];
        }
        return v_index;
}

inline size_t Tensor::getSize(TensorShape dims) const {
        int size = 1;
        for (int i = 0; i < shape_.size(); ++i) {
                size *= shape_[i];
        }
        return size;
}

float& Tensor::operator[](TensorShape dims)
{
        if (dims.size() != shape_.size()) {
                throw std::out_of_range("Tensor index out of range");
        }

        return data_[this->getIndex(dims)];
}

const float Tensor::operator[](TensorShape dims) const
{
        if (dims.size() != shape_.size()) {
                throw std::out_of_range("Tensor index out of range");
        }

        return data_[this->getIndex(dims)];
}

const bool Tensor::operator==(const Tensor& other) const
{
        return this->data_ == other.vector();
}

const bool Tensor::precisionEqual(const Tensor& other, const int precision) const
{
        auto ours = data_;
        auto theirs = other.vector();
        float p = pow(10, -p);
        bool correct = true;
        for (int i = 0; i < data_.size(); i++) {
                bool correct = std::fabs(ours[i] - theirs[i]) < p;
                if (!correct) {
                        return false;
                }
        }
        return true;
}

const TensorShape Tensor::shape() const {
        return shape_;
}

const long unsigned int Tensor::shape(int idx) const {
        return shape_[idx];
}

float const * Tensor::data() const {
        return data_.data();
}

const std::vector<float>& Tensor::vector() const {
        return data_;
}

void Tensor::fill(const float& value) {
    std::fill(data_.begin(), data_.end(), value);
}

std::string Tensor::shapeString() const {
        std::ostringstream os;
        os << "(";
        for (int i = 0; i < shape_.size(); ++i) {
                os << shape_[i]<< ",";
        }
        os << ")";
        return os.str();
}
