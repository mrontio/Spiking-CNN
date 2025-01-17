#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cnpy.h>
#include <string>
#include <iostream>
#include <numeric>
#include <sstream>
#include <cmath>
#include <format>
#include <limits>

using shape = long unsigned int;
using TensorShape = std::vector<shape>;

class Tensor {
private:
        TensorShape shape_;
        std::unique_ptr<std::vector<float>> data_;


        size_t getIndex(const TensorShape& dims) const;


public:
        int count;

        Tensor();
        Tensor(const TensorShape& dims);
        Tensor(const std::string);
        Tensor(const cnpy::NpyArray &npy);
        Tensor(const Tensor& source);
        Tensor(const float* source, const TensorShape& shape);

        static size_t getSize(const TensorShape& shape);
        static TensorShape shave(const TensorShape& shape, const int n);

        float& operator[](const TensorShape& dims);
        //const float operator[](const TensorShape& dims);

        std::unique_ptr<Tensor> operator()(const TensorShape& shape);

        const TensorShape shape() const ;
        const long unsigned int shape(int idx) const;
        const size_t size() const;

        Tensor& reshape(const TensorShape& shape);
        Tensor& flatten();
        Tensor& flatten(int dim);

        std::string shapeString () const;
        void fill(const float& value);

        const bool operator==(const Tensor& other) const;
        const bool precisionEqual(const Tensor& other, const int precision) const;

        // For the following, we assume that shape = {100, 10}
        std::unique_ptr<Tensor> sum() ;
        int argmax() const;

        const float* data() const;
        const std::vector<float>& vector() const;

        void save(std::string filename) const;
        void fillDebug();


};

#include "Tensor.tpp"

#endif // TENSOR_H
