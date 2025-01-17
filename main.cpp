#include <iostream>
#include <format>
#include <chrono>

//#define DEBUG_WRITE

#include "Tensor.h"
#include "Convolutional.h"
#include "IntegrateFire.h"
#include "AvgPool.h"
#include "Linear.h"
#include <filesystem>

using namespace std::chrono;
using namespace std;

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
        int exec_ms_time_sum = 0;
        int count = 0;
        int total = 10000;

        string data_path = "/home/mrontio/data/nmnist-converted/";
        for (const auto& class_directory_dir : std::filesystem::directory_iterator(data_path)) {
                const auto& class_directory = class_directory_dir.path().string();
                const int correct_class = int(class_directory[class_directory.size() - 1]) - 48;

                for (const auto& data_path_entry : std::filesystem::directory_iterator(class_directory)) {
                        // Load Data
                        const auto& data_path = data_path_entry.path().string();
                        auto time_start = high_resolution_clock::now();
                        auto data = make_unique<Tensor>(data_path);

                        // Latent spaces
                        std::vector<std::unique_ptr<Tensor>> xs = std::vector<std::unique_ptr<Tensor>>(11);

                        // We run the network
                        xs[0] = c0.forward(*data);
                        xs[1] = i1.forward(*(xs[0]));
                        xs[2] = a2.forward(*(xs[1]));
                        xs[3] = c3.forward(*(xs[2]));
                        xs[4] = i4.forward(*(xs[3]));
                        xs[5] = a5.forward(*(xs[4]));
                        xs[6] = c6.forward(*(xs[5]));
                        xs[7] = i7.forward(*(xs[6]));
                        xs[7]->flatten(1);
                        xs[8] = l9.forward(*(xs[7]));
                        xs[9] = i10.forward(*(xs[8]));
                        xs[10] = xs[9]->sum();
                        int pred = xs[10]->argmax();

                        correct += correct_class == pred;
                        ++count;
                        auto time_end = high_resolution_clock::now();
                        exec_ms_time_sum += duration_cast<milliseconds>(time_end - time_start).count();
                        int avg_exec_ms_time = exec_ms_time_sum / count;
                        int pred_total_m_time = ((avg_exec_ms_time * total) / 1000) / 60;

                        printf("(%d/%d) accuracy: %.2f%% (%d/%d), avg_exec_ms_time: %d, pred_total_s_time: %d \n",
                               count, total, ((float)correct)/((float)count) * 100.0, correct, count,
                               avg_exec_ms_time, pred_total_m_time);

                        return 0;

                }
                printf("\nTest complete\nAccuracy: %.2f (%d/%d)\n",
                       (float)correct/(float)count, correct, count);
        }

        return 0;
}
