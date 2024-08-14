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
        auto mine = conv2d.forward(x);

        cnpy::NpyArray theirs_npy = cnpy::npy_load("./theirs.npy");
        cnpy::NpyArray mine_npy = cnpy::npy_load("./mine.npy");
        auto torch = Tensor(theirs_npy);
        auto py = Tensor(mine_npy);

        auto idx = TensorShape{0,0,0};
        cout << torch[idx] << endl;
        cout << py[idx] << endl;
        cout << (*mine)[idx] << endl;

        cout << (mine->precisionEqual(torch, 5)) << endl;

        return 0;
}
