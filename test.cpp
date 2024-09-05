#include <iostream>
#include <vector>
#include <memory>

#include "Tensor.h"
#include "Convolutional.h"
#include "IntegrateFire.h"

using namespace std;

#define DEBUG_WRITE
#ifdef DEBUG_WRITE
#define DEBUG_WRITE_CMD x->save("./tensors/layer" + to_string(count) + ".npy"); printf("Wrote ./tensors/layer/%d.npy\n", count++);
#else
#define DEBUG_WRITE_CMD
#endif

int main(int argc, char *argv[])
{
        auto data = make_unique<Tensor>(TensorShape{2, 34, 34});
        data->fill(0.2f);

        int count = 0;
        auto c0 = Convolutional({2, 34, 34}, Tensor("./weights/0-Conv2d.npy"), 1, {1, 1});
        auto i1 = IntegrateFire(TensorShape{8, 34, 34});

        auto x = c0.forward(*data);
        DEBUG_WRITE_CMD;
        auto d = *x;
        for (int i = 0; i < 5; ++i) {
                x = i1.forward(d);
                DEBUG_WRITE_CMD;
        }

        return 0;
}
