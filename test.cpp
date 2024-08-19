#include <iostream>
#include <vector>

#include "Tensor.h"
#include "Linear.h"

using namespace std;

int main(int argc, char *argv[])
{
        auto linear = Linear(Tensor("./weights/9-Linear.npy"));
        Tensor x = Tensor(TensorShape{256});
        x.fill(1.0f);

        auto out = linear.forward(x);

        return 0;
}
