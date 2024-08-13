#include <iostream>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <cnpy.h>

using namespace std;

#include "Tensor.h"


Tensor::Tensor()
        : shape_(TensorShape()),
          data_(),
          N_(0)
{}

Tensor::Tensor(TensorShape dims)
        : shape_(dims),
          N_(dims.size())
{
        int size = 1;
        for (int n = 0; n < N_; n++) {
                size *= dims[n];
        }
        data_ = vector<float>(size);
}

Tensor::Tensor(const cnpy::NpyArray& npy)
        : shape_(npy.shape),
        N_(npy.shape.size())
{
        int size = 1;
        for (int i = 0; i < N_; ++i) {
                size *= shape_[i];
        }
        const float* elements = npy.data<float>();
        data_ = std::vector<float>(elements, elements + size );
}

Tensor::Tensor(const Tensor& source)
        : shape_(source.getShape()),
          N_(source.getShape().size())
{
        const float* data = source.data();
        data_= std::vector<float>(data, data+ (shape_[0] * shape_[1] * shape_[2] * shape_[3]));
}




size_t Tensor::getIndex(TensorShape dims) const
{
        int v_index = 0;
        for (int i = 0; i < N_; ++i) {
                v_index += dims[i] * shape_[i];
        }
        return v_index;
}

float& Tensor::operator[](TensorShape dims)
{
        if (dims.size() != N_) {
                throw std::out_of_range("Tensor index out of range");
        }

        return data_[this->getIndex(dims)];
}

const float Tensor::operator[](TensorShape dims) const
{
        if (dims.size() != N_) {
                throw std::out_of_range("Tensor index out of range");
        }

        return data_[this->getIndex(dims)];
}

TensorShape Tensor::shape() const {
        return shape_;
}

float const * Tensor::data() const {
        return data_.data();
}

void Tensor::fill(const float& value) {
    std::fill(data_.begin(), data_.end(), value);
}

std::string Tensor::shapeString() const {
        std::ostringstream os;
        os << "(";
        for (int i = 0; i < N_; ++i) {
                os << shape_[i]<< ",";
        }
        os << ")";
        return os.str();
}
