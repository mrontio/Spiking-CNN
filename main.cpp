#include <iostream>

#include "Tensor.h"
#include "Convolutional.h"
#include "IntegrateFire.h"
#include "AvgPool.h"
#include "Linear.h"

using namespace std;

int main() {
        string filename = "0.npy";
        auto data = make_unique<Tensor>("/home/mrontio/data/nmnist-converted/0/" + filename);

        auto input_shape = TensorShape{2, 34, 34};

        auto b0_shape = TensorShape{8, 34, 34};
        auto c0 = Convolutional(input_shape, Tensor("./weights/0-Conv2d.npy"), 1, {1, 1});
        auto i1 = IntegrateFire(b0_shape);
        auto a2 = AvgPool(b0_shape, 2, 2, 0);

        auto b1_shape = TensorShape{16, 17, 17};
        auto c3 = Convolutional(b1_shape, Tensor("./weights/3-Conv2d.npy"), 1, {1, 1});
        auto i4 = IntegrateFire(b1_shape);
        auto a5 = AvgPool(b1_shape, 2, 2, 0);

        auto b2_shape = TensorShape{15, 8, 8};
        auto c6 = Convolutional(b2_shape, Tensor("./weights/6-Conv2d.npy"), 2, {1, 1});
        auto i7 = IntegrateFire(b2_shape);
        // Step 8 missing: we flatten the tensor in-place.
        auto l9 = Linear(Tensor("./weights/9-Linear.npy"));
        // Not yet, my IF currently expects a dimensionality of 3
                // auto i10 = IntegrateFire(TensorShape{10});

        int correct = 0;
        for (shape batch = 0; batch < 1 /*data->shape(0)*/; batch++) {

                auto x = (*data)(TensorShape{batch});
                x = c0.forward(*x);
                x = i1.forward(*x);
                x = a2.forward(*x);
                x = c3.forward(*x);
                x = i4.forward(*x);
                x = a5.forward(*x);
                x = c6.forward(*x);
                x = i7.forward(*x);
                x->flatten();
                x = l9.forward(*x);
                // i10 once you've implemented dynamic dimensions

                x->save("./tensors/" + filename);
                auto pred = x->argmax();
                correct += pred == 0;
                cout << pred << endl;
        }
        cout << correct << endl;
        return 0;
}
