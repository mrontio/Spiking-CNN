#include <iostream>

#include "Tensor.h"
#include "Convolutional.h"
#include "IntegrateFire.h"
#include "AvgPool.h"

using namespace std;

int main() {
        auto conv_weights = Tensor("./weights/conv-weights.npy");
        auto torch_output = Tensor("./tensors/pool-torch.npy");

        auto data = Tensor("./tensors/sample-data.npy");
        auto x = Tensor(TensorShape{2,34,34});
        x.fill(1.0);

        auto conv2d = Convolutional(conv_weights, 1, {1, 1});
        auto conv_out = conv2d.forward(x);

        auto if_layer = IntegrateFire((*conv_out).shape());
        auto if_out = if_layer.forward((*conv_out));

        auto avgpool = AvgPool(2, 2, 0);
        auto pool_out = avgpool.forward(*if_out);


        cout << pool_out->precisionEqual(torch_output, 8) << endl;

        return 0;
}
