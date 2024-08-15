// test.cpp
//#include <cnpy.h>
#include <iostream>

#include "Tensor.h"
#include "Convolutional.h"
#include "IntegrateFire.h"

using namespace std;

int main() {
        cnpy::NpyArray data = cnpy::npy_load("./weights/conv-weights.npy");
        auto conv_weights = Tensor(data);
        auto x = Tensor(TensorShape{8,2,34,34});
        x.fill(1.0);

        auto conv2d = Convolutional(conv_weights, 1, {1, 1});
        auto conv_out = conv2d.forward(x);

        auto if_layer = IntegrateFire((*conv_out).shape());
        auto if_out = if_layer.forward((*conv_out));

        auto torch_output = Tensor(cnpy::npy_load("./tensors/if-torch.npy"));

        if_out->flatten();

        cout << if_out->shapeString() << endl;

        return 0;
}
