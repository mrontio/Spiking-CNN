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
        int count = 0;
        int total = 10000;

        string data_path = "/home/mrontio/data/nmnist-converted/";
        for (const auto& class_directory_dir : std::filesystem::directory_iterator(data_path)) {
                const auto& class_directory = class_directory_dir.path().string();
                const int correct_class = int(class_directory[class_directory.size() - 1]) - 48;
                cout << class_directory << endl;
                for (const auto& data_path_entry : std::filesystem::directory_iterator(class_directory)) {
                        // Load Data
                        const auto& data_path = data_path_entry.path().string();
                        auto data = make_unique<Tensor>(data_path);


                        // We run the network
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
                        x = x->sum();
                        int pred = x->argmax();

                        correct += correct_class == pred;
                        ++count;
                        printf("(%d/%d) Accuracy: %.2f%% (%d/%d)\n", count, total,
                               ((float)correct)/((float)count) * 100.0, correct, count);


                }
                printf("\nTest complete\nAccuracy: %.2f (%d/%d)\n",
                       (float)correct/(float)count, correct, count);
        }

        return 0;
}
