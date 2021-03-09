#pragma once

#include <iostream>

#include "../Tensor/tensor.cuh"

class Layer {
public:
    virtual Tensor& forward(Tensor& input) = 0;
    virtual Tensor& backward(Tensor& gradients, float lr) = 0;

    virtual void info() = 0;
};