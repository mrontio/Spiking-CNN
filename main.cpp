// test.cpp
//#include <cnpy.h>
#include <iostream>

//#include "conv.hpp"
#include "Tensor4D.h"
#include "Convolutional.h"

using namespace std;

int main() {
        cnpy::NpyArray data = cnpy::npy_load("./weights.npy");
        auto conv_weights = Tensor4D{data};
        auto x = Tensor4D(8,2,34,34);
        x.fill(1.0);

        auto conv2d = Convolutional(conv_weights, 1, vector<int>{1, 1});
        conv2d.forward(x);




        return 0;
}
