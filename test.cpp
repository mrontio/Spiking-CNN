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
        auto s = TensorShape{100, 8, 34, 34};
        auto data = make_unique<Tensor>(s);
        data->fill(0.2f);

        auto c0 = Convolutional(data->shape(), Tensor("./weights/0-Conv2d.npy"), 1, {1, 1});
        auto i1 = IntegrateFire(s);

        auto x = c0.forward(*data);
        x = i1.forward(*x);
        cout << x->shapeString() << endl;
        x->save("./lol.npy");


        return 0;
}
