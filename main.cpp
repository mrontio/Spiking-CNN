#include <iostream>

#include "Tensor.h"
#include "Convolutional.h"
#include "IntegrateFire.h"
#include "AvgPool.h"

using namespace std;

int main() {
        auto data = Tensor("./tensors/sample-data.npy");
        //auto torch_output = Tensor("./tensors/output-3-AvgPool2d.npy");
        auto out_conv2d = Tensor("./tensors/output-0-Conv2d.npy");
        auto out_iaf = Tensor("./tensors/output-1-IAFSqueeze.npy");
        auto recordings = Tensor("./tensors/recording-1-IAFSqueeze.npy");

        auto conv2d = Convolutional(Tensor("./weights/0-Conv2d.npy"), 1, {1, 1});
        auto if_layer = IntegrateFire(TensorShape{8, 34, 34});
        auto avgpool = AvgPool(2, 2, 0);

        int correct = 0;
        for (shape batch = 0; batch < data.shape(0); batch++) {
                auto x = data(TensorShape{batch});
                auto y = out_conv2d(TensorShape{batch});

                auto conv_out = conv2d.forward(*x);
                auto if_out = if_layer.forward(*conv_out);
                //auto pool_out = avgpool.forward(*if_out);

                bool cool = conv_out->precisionEqual(*y, 6);
                if (cool) {
                        ++correct;
                }
                cout << batch << " " << correct << endl;
        }
        cout << "Correct: " << correct << ", Wrong: " << data.shape(0) - correct << endl;

        return 0;
}
