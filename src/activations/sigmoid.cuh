#pragma once

#include "../layers/layer.hpp"

class Sigmoid : public Layer {
public:
    Sigmoid() = default;
    ~Sigmoid() = default;

    Tensor& forward(Tensor& Z);
    Tensor& backward(Tensor& dA, float lr = 0.01);

    void info();

private:
    Tensor A;

    Tensor Z;
    Tensor dZ;
};
