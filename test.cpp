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
        auto s = TensorShape{100, 2, 34, 34};
        auto data = make_unique<Tensor>(s);
        data->fill(0.2f);

        auto c0 = Convolutional(s, Tensor("./weights/0-Conv2d.npy"), 1, {1, 1});
        auto x = c0.forward(*data);
        x->save("./lol.npy");
        // DEBUG_WRITE_CMD;
        // auto d = *x;
        // for (int i = 0; i < 5; ++i) {
        //         x = i1.forward(d);
        //         DEBUG_WRITE_CMD;
        // }

        return 0;
}
