// test.cpp
//#include <cnpy.h>
#include <iostream>

#include "Tensor.h"
#include "Convolutional.h"

using namespace std;

int main() {
        cnpy::NpyArray data = cnpy::npy_load("./weights.npy");
        auto conv_weights = Tensor(data);
        auto x = Tensor(TensorShape{8,2,34,34});
        x.fill(1.0);

        auto conv2d = Convolutional(conv_weights, 1, {1, 1});
        // auto conv2d.forward(x);



        return 0;
}
