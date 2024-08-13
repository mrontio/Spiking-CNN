// test.cpp
//#include <cnpy.h>
#include <iostream>

#include "Tensor4D.h"
#include "Tensor.h"
// #include "Convolutional.h"

using namespace std;

int main() {
        cnpy::NpyArray data = cnpy::npy_load("./weights.npy");
        auto conv_weights_4d = Tensor4D(data);
        auto conv_weights = Tensor(data);
        auto d1 = conv_weights.data();
        auto d2 = conv_weights_4d.data();
        cout << conv_weights.shapeString() << endl;
        cout << conv_weights[TensorShape{0,0,0,0}] << endl;

        cout << conv_weights_4d.shapeString() << endl;
        cout << conv_weights_4d(0,0,0,1) << endl;
        cout << d1[0] << endl;
        cout << d2[0] << endl;
        // auto x = Tensor4D(8,2,34,34);
        // x.fill(1.0);

        // auto conv2d = Convolutional(conv_weights, 1, vector<int>{1, 1});
        // auto conv2d.forward(x);



        return 0;
}
