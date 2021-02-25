#pragma once

#include "layer.hpp"

class Sigmoid : public Layer {
public:
    Sigmoid(std::string name);
    ~Sigmoid() = default;

    Tensor& forward(Tensor& Z);
    Tensor& backward(Tensor& dA, float lr = 0.01);
private:
    Tensor A;

    Tensor Z;
    Tensor dZ;
};
