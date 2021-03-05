#pragma once

#include "../layers/layer.hpp"

class ReLU : public Layer {
public:
    ReLU(std::string name);
    ~ReLU() = default;

    Tensor& forward(Tensor& Z);
    Tensor& backward(Tensor& dA, float lr = 0.01);

private:
    Tensor A;

    Tensor Z;
    Tensor dZ;
};
