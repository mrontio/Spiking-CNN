#include <iostream>
#include <vector>

#include "Tensor.h"

using namespace std;

int main(int argc, char *argv[])
{
        Tensor t = Tensor(TensorShape{8, 3, 4, 4});
        t[TensorShape{0,0,0,0}] = 5;
        cout << t[TensorShape{0,0,0,0}] << endl;

        return 0;
}
