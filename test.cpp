#include <iostream>
#include <vector>

#include "Tensor.h"

using namespace std;

int main(int argc, char *argv[])
{
        Tensor t = Tensor(TensorShape{8, 3, 4, 4});
        t.fillDebug();

        auto t1 = t(TensorShape{7, 2, 3, 3});


        return 0;
}
