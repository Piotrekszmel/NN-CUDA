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
    net.addLayer(new Linear("linear_1", Shape(2, 30)));
    net.addLayer(new ReLU("relu_1"));
    net.addLayer(new Linear("linear_2", Shape(30, 1)));
    net.addLayer(new Sigmoid("sigmoid_output"));
    
    net.train(100, 21, 1000);

    return 0;
}
