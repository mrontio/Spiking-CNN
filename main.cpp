#include <iostream>
#include <format>

//#define DEBUG_WRITE

#include "Tensor.h"
#include "Convolutional.h"
#include "IntegrateFire.h"
#include "AvgPool.h"
#include "Linear.h"
#include <filesystem>




using namespace std;


#ifdef DEBUG_WRITE
#define DEBUG_WRITE_X_FILE ("./tensors/x" + to_string(count) + ".npy").c_str()
#define DEBUG_WRITE_X_CMD x->save(DEBUG_WRITE_X_FILE); printf("Wrote %s\n", DEBUG_WRITE_X_FILE); ++count;
#define DEBUG_WRITE_MEMBRANE_FILE ("./tensors/membrane" + to_string(count) + ".npy").c_str()
#define DEBUG_WRITE_MEMBRANE_CMD x->save(DEBUG_WRITE_MEMBRANE_FILE); printf("Wrote %s\n", DEBUG_WRITE_MEMBRANE_FILE);
#else
#define DEBUG_WRITE_X_CMD
#define DEBUG_WRITE_MEMBRANE_FILE
#endif

int main() {
        int real = 2;
        string filename = to_string(real) + "200.npy";
        auto data = make_unique<Tensor>("/home/mrontio/data/nmnist-converted/" + to_string(real) + "/"  + filename);

#ifdef DEBUG_WRITE
        filesystem::remove_all("./tensors/");
        std::filesystem::create_directory("./tensors");

#endif
        auto input_shape = TensorShape{100, 2, 34, 34};

        auto b0_shape = TensorShape{100, 8, 34, 34};
        auto c0 = Convolutional(input_shape, Tensor("./weights/0-Conv2d.npy"), 1, {1, 1});
        auto i1 = IntegrateFire(b0_shape);
        auto a2 = AvgPool(b0_shape, 2, 2, 0);

        auto b1_shape = TensorShape{100, 16, 17, 17};
        auto c3 = Convolutional(b1_shape, Tensor("./weights/3-Conv2d.npy"), 1, {1, 1});
        auto i4 = IntegrateFire(b1_shape);
        auto a5 = AvgPool(b1_shape, 2, 2, 0);

        auto b2_shape = TensorShape{100, 16, 4, 4};
        auto c6 = Convolutional(TensorShape{100, 16, 8, 8}, Tensor("./weights/6-Conv2d.npy"), 2, {1, 1});
        auto i7 = IntegrateFire(b2_shape);
        // Step 8 missing: we flatten the tensor in-place.
        auto l9 = Linear(Tensor("./weights/9-Linear.npy"));
        auto i10 = IntegrateFire(TensorShape{100, 10});

        int correct = 0;
        int datacount = 0;
        int count = 0;
        auto mem = i1.membrane_;

        int batches = 1;
        for (shape b = 0; b < batches; b++) {
                data->save("./tensors/data.npy");

                auto x = c0.forward(*data);
                x = i1.forward(*x);
                x = a2.forward(*x);
                x = c3.forward(*x);
                x = i4.forward(*x);
                x = a5.forward(*x);
                x = c6.forward(*x);
                x = i7.forward(*x);
                x->flatten(1);
                x = l9.forward(*x);
                x = i10.forward(*x);
                x->save("./tensors/out.npy");

                x = x->sum();
                cout << "Prediction: " << x->argmax() << endl;
        }

        return 0;
}
