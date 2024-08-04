// test.cpp
//#include <cnpy.h>
#include <iostream>

//#include "conv.hpp"
#include "Tensor4D.h"
using namespace std;



int main() {
        cnpy::NpyArray data = cnpy::npy_load("./weights.npy");
        auto t = Tensor4D<float>{data};
        cout << t.shapeString() << endl;
        cout << t(0,0,0,0) << endl;
        cout << t(0,1,0,1) << endl;
        cout << t(7,1,2,2) << endl;
        return 0;
}
