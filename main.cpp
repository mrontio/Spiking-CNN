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
        int real = 6;
        string filename = to_string(real) + "172.npy";
        // auto data = make_unique<Tensor>("/home/mrontio/data/nmnist-converted/" + to_string(real) + "/"  + filename);
        auto data = make_unique<Tensor>("./6174.npy");
        //shape total_count = 5;
        //auto data = make_unique<Tensor>(TensorShape{total_count, 2, 34, 34});
        //data->fill(0.2f);
        //data->fillDebug();

#ifdef DEBUG_WRITE
        filesystem::remove_all("./tensors/");
        std::filesystem::create_directory("./tensors");

#endif
        //data->save("./tensors/" + filename);
        //cout << "Wrote ./tensors/" << filename <<  endl;

        auto input_shape = TensorShape{2, 34, 34};

        auto b0_shape = TensorShape{8, 34, 34};
        auto c0 = Convolutional(input_shape, Tensor("./weights/0-Conv2d.npy"), 1, {1, 1});
        auto i1 = IntegrateFire(b0_shape);
        auto a2 = AvgPool(b0_shape, 2, 2, 0);

        auto b1_shape = TensorShape{16, 17, 17};
        auto c3 = Convolutional(b1_shape, Tensor("./weights/3-Conv2d.npy"), 1, {1, 1});
        auto i4 = IntegrateFire(b1_shape);
        auto a5 = AvgPool(b1_shape, 2, 2, 0);

        auto b2_shape = TensorShape{16, 4, 4};
        auto c6 = Convolutional(b2_shape, Tensor("./weights/6-Conv2d.npy"), 2, {1, 1});
        auto i7 = IntegrateFire(b2_shape);
        // Step 8 missing: we flatten the tensor in-place.
        auto l9 = Linear(Tensor("./weights/9-Linear.npy"));
        auto i10 = IntegrateFire(TensorShape{10});

        int correct = 0;
        int datacount = 0;
        int count = 0;
        auto mem = i1.membrane_;

        auto pred = Tensor(TensorShape(10));

        for (shape t = 0; t < data->shape(0); t++) {
                auto x = (*data)(TensorShape{t});

                x = c0.forward(*x);
                DEBUG_WRITE_X_CMD;
                x = i1.forward(*x);
                // i1.membrane_.save(DEBUG_WRITE_MEMBRANE_FILE); printf("Wrote %s\n", DEBUG_WRITE_MEMBRANE_FILE);
                DEBUG_WRITE_X_CMD;
                x = a2.forward(*x);
                DEBUG_WRITE_X_CMD;
                x = c3.forward(*x);
                DEBUG_WRITE_X_CMD;
                x = i4.forward(*x);
                // i4.membrane_.save(DEBUG_WRITE_MEMBRANE_FILE); printf("Wrote %s\n", DEBUG_WRITE_MEMBRANE_FILE);
                DEBUG_WRITE_X_CMD;
                x = a5.forward(*x);
                DEBUG_WRITE_X_CMD;
                x = c6.forward(*x);
                DEBUG_WRITE_X_CMD;
                x = i7.forward(*x);
                // i7.membrane_.save(DEBUG_WRITE_MEMBRANE_FILE); printf("Wrote %s\n", DEBUG_WRITE_MEMBRANE_FILE);
                DEBUG_WRITE_X_CMD;
                x->flatten();
                DEBUG_WRITE_X_CMD;
                x = l9.forward(*x);
                DEBUG_WRITE_X_CMD;
                x = i10.forward(*x);
                // i10.membrane_.save(DEBUG_WRITE_MEMBRANE_FILE); printf("Wrote %s\n", DEBUG_WRITE_MEMBRANE_FILE);
                DEBUG_WRITE_X_CMD;

                cout << x->argmax() << endl;

        }
        //cout << pred.argmax() << endl;
        return 0;
}
