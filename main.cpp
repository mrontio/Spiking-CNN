// test.cpp
//#include <cnpy.h>
#include <iostream>

//#include "conv.hpp"
#include "Tensor4D.h"
using namespace std;



// int main() {
//         cnpy::NpyArray data = cnpy::npy_load("./weights.npy");
//         for (int i = 0; i < data.shape.size(); i++) {

//                 cout << data.shape[i] << ",";

//         }
//         cout << endl;
//         vector* weights = data.data<vector>();
//         cout << (*weights)[0][0][0][0] << endl;

//         return 0;
// }

int main() {
        // Create a 4x3x2x2 tensor
        Tensor4D<float> tensor(4, 3, 2, 2);

        // Fill the tensor with a value
        tensor.fill(1.0f);

        // Access and modify elements
        tensor(0, 0, 0, 0) = 5.0f;
        tensor(1, 1, 1, 1) = 10.0f;

        // Print some elements
        std::cout << "tensor(0, 0, 0, 0): " << tensor(0, 0, 0, 0) << std::endl;
        std::cout << "tensor(1, 1, 1, 1): " << tensor(1, 1, 1, 1) << std::endl;

        return 0;
}
