#pragma once

#include <iostream>

#include "../Tensor/tensor.cuh"

class Layer {
public:
    virtual ~Layer() = 0;

    virtual Tensor& forward(Tensor& input) = 0;
    virtual Tensor& backward(Tensor& gradients, float lr) = 0;

    std::string getName() { return this->name; }

protected:
    std::string name;
};