#include <iostream>
#include <vector>
#include <memory>

#include "Tensor.h"
#include "Convolutional.h"
#include "IntegrateFire.h"
#include "AvgPool.h"
#include "Linear.h"

using namespace std;

//#define DEBUG_WRITE
#ifdef DEBUG_WRITE
#define DEBUG_WRITE_CMD x->save("./tensors/layer" + to_string(count) + ".npy"); printf("Wrote ./tensors/layer/%d.npy\n", count++);
#else
#define DEBUG_WRITE_CMD
#endif

int main(int argc, char *argv[])
{
        auto s = TensorShape{};
        auto data = make_unique<Tensor>(TensorShape{100, 16, 17, 17});
        // data->fillDebug();
        // data->save("./please.npy");

        // auto l = Linear(Tensor("./weights/9-Linear.npy"));
        // auto x = l.forward(*data);
        data->flatten(2);
        cout << data->shapeString() << endl;
        //x->save("./lol.npy");


        return 0;
}
