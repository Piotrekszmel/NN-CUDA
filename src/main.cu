#include <iostream>
#include <ctime>

#include "Net/net.cuh"
#include "layers/linear.cuh"
#include "activations/relu.cuh"
#include "activations/sigmoid.cuh"
#include "loss/binary_cross_entropy.cuh"

#include "Tensor/tensor.cuh"

float computeAccuracy(const Tensor& y_pred, const Tensor& y_true);

int main()
{
    srand(time(NULL));

    Net net;
    net.addLayer(new Linear(Shape(2, 30)));
    net.addLayer(new ReLU());
    net.addLayer(new Linear(Shape(30, 1)));
    net.addLayer(new Sigmoid());
    
    net.train(100, 21, 1000);

    return 0;
}
