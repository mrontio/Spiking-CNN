// test.cpp
#include <cnpy.h>
#include <iostream>

//#include "conv.hpp"

int main() {
        cnpy::NpyArray data = cnpy::npy_load("./weights.npy");
        float* loaded_data = data.data<float>();
        printf("%f\n", loaded_data[0]);
        return 0;
}
