#include "Tensor.h"

using namespace std;

Tensor::Tensor()
        : shape_(TensorShape()),
          data_()
{}

Tensor::Tensor(const std::string path)
{
        auto npy = cnpy::npy_load(path);
        const float* elements = npy.data<float>();

        shape_ = npy.shape;
        data_ = make_unique<std::vector<float>>(elements, elements + getSize(shape_));
}

Tensor::Tensor(const TensorShape& dims)
        : shape_(dims)
{
        int size = 1;
        for (int n = 0; n < shape_.size(); n++) {
                size *= dims[n];
        }
        data_ = make_unique<std::vector<float>>(size);
}


Tensor::Tensor(const cnpy::NpyArray& npy)
        : shape_(npy.shape)
{
        const float* elements = npy.data<float>();
        data_ = make_unique<std::vector<float>>(elements, elements + getSize(shape_));
}

Tensor::Tensor(const Tensor& source)
        : shape_(source.shape())
{
        const float* data = source.data();
        data_= make_unique<std::vector<float>>(data, data + getSize(shape_));
}

Tensor::Tensor(const float* source, const TensorShape& shape)
        : shape_(shape)
{
        data_ = make_unique<std::vector<float>>(source, source + getSize(shape));
}

Tensor& Tensor::reshape(const TensorShape& shape)
{
        auto size = getSize(shape);
        if (size != this->size()) {
                throw std::runtime_error("shape : element mismatch");
        }
        shape_ = shape;
        return *this;
}

Tensor& Tensor::flatten() {
        auto size = getSize(shape_);
        shape_.assign({getSize(shape_)});
        return *this;
}

size_t Tensor::getIndex(const TensorShape& dims) const
{
        int v_index = dims[0];
        for (int i = 1; i < shape_.size(); ++i) {
                v_index = v_index * shape_[i] + dims[i];
        }
        return v_index;
}



float& Tensor::operator[](const TensorShape& dims)
{
        if (dims.size() != shape_.size()) {
                throw std::out_of_range("Tensor index out of range");
        }

        return (*data_)[this->getIndex(dims)];
}

const float Tensor::operator[](const TensorShape& dims) const
{
        if (dims.size() != shape_.size()) {
                throw std::out_of_range("Tensor index out of range");
        }

        return (*data_)[this->getIndex(dims)];
}

/**
   Sub-tensor operator.
   This returns a a pointer to a sub-tensor.
   @shape dimensions must be less than that of the parent tensor.
 */
std::unique_ptr<Tensor> Tensor::operator()(const TensorShape& shape) {
        auto dim_n = shape.size();
        auto dim_diff =  shape_.size() - dim_n;
        if (dim_diff < 0) {
                throw std::out_of_range("Sub-tensor dimensions are bigger than parent tensor!");
        }

        auto begin = shape;
        auto end =  shape;
        auto new_shape = TensorShape{};
        for (int i = dim_n; i < shape_.size(); ++i) {
                begin.emplace_back(0);
                end.emplace_back(shape_[i] - 1);
                new_shape.emplace_back(shape_[i]);
        }
        if (dim_diff == 0) new_shape = TensorShape{1};

        float* beginp = data_->data() + getIndex(begin);
        auto output = make_unique<Tensor>(beginp, new_shape);

        return output;

}

const bool Tensor::operator==(const Tensor& other) const
{
        return *this->data_ == other.vector();
}

const bool Tensor::precisionEqual(const Tensor& other, const int precision) const
{
        if (data_->size() != other.size()) {
                cout << "precisionEqual: size mismatch " << data_->size() << ", " << other.size() << endl;
                return false;
        }
        auto ours = *data_;
        auto theirs = other.vector();
        float p = float(pow(10, -precision));
        bool correct = true;
        for (int i = 0; i < data_->size(); i++) {
                bool correct = std::fabs(ours[i] - theirs[i]) < p;
                if (!correct) {
                        return false;
                }
        }
        return true;
}

// Assume shape = {100, 10}
std::unique_ptr<Tensor> Tensor::sum() const {
        int axes = shape_[0];
        auto out = make_unique<Tensor>(TensorShape{shape_[1]});
        out->fill(0.0f);
        for (long unsigned int i = 0; i < shape_[1]; ++i) {
                for (long unsigned int j = 0; j < axes; ++j) {
                        auto idx = TensorShape{j, i};
                        (*out)[TensorShape{i}] += (*this)[idx];
                }
        }
        return out;
}

// Assume shape = {10}
int Tensor::argmax() const {
        float max = - std::numeric_limits<float>::max();
        int argmax = -1;
        for (long unsigned int i = 0; i < shape_[0]; ++i) {
                auto data = (*data_)[i];
                cout << data;
                if (data > max) {
                        max = data;
                        argmax = i;
                }
        }
        cout << ":";
        return argmax;
}

const TensorShape Tensor::shape() const {
        return shape_;
}

const long unsigned int Tensor::shape(int idx) const {
        return shape_[idx];
}

const size_t Tensor::size() const {
        return data_->size();
}

const size_t Tensor::getSize(const TensorShape& shape) const
{
        int size = 1;
        for (int i = 0; i < shape.size(); ++i) {
                size *= shape[i];
        }
        return size;
}

float const * Tensor::data() const {
        return data_->data();
}

const std::vector<float>& Tensor::vector() const {
        return *data_;
}

void Tensor::fill(const float& value) {
    std::fill(data_->begin(), data_->end(), value);
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

void Tensor::fillDebug()
{
        for (int i = 0; i < data_->size(); i++) {
                (*data_)[i] = i;
        }
}

void Tensor::save(string filepath) const {
        cnpy::npy_save(filepath, data_->data(), shape_, "w");
}
