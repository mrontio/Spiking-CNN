#include <iostream>
#include <vector>
#include <memory>

#include "Tensor.h"

using namespace std;

int main(int argc, char *argv[])
{
        //auto data = std::make_unique<Tensor>("/home/mrontio/data/nmnist-converted/0/0.npy");
        auto data = std::make_unique<Tensor>(TensorShape{100, 10});
        data->fill(1.0f);
        auto sum = data->sum();
        cout << sum->shapeString() << endl;
        for (shape i = 0; i < 10; ++i) {
                cout << (*sum)[TensorShape{i}] << " ";
        }
        cout << endl;
        (*sum)[TensorShape{0}] = 0;
        cout << sum->argmax() << endl;
        return 0;
}
