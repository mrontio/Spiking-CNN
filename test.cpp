#include <iostream>
#include <vector>
#include <memory>

#include "Tensor.h"
o#include "Convolutional.h"
// #include "IntegrateFire.h"
// #include "AvgPool.h"
// #include "Linear.h"

using namespace std;

int main(int argc, char *argv[])
{
        auto data = make_unique<Tensor>("/home/mrontio/data/nmnist-converted/0/593.npy");
        auto input_shape = TensorShape{100, 2, 34, 34};
        auto c0 = Convolutional(input_shape, Tensor("./weights/0-Conv2d.npy"), 1, {1, 1});
        auto x = c0.forward(*data);

        return 0;
}
