#pragma once

#include "layer.hpp"

class Linear : public Layer {
public:
    Linear(std::string name, Shape W_shape);
    ~Linear() = default;

    Tensor& forward(Tensor& input);
    Tensor& backward(Tensor& gradients, float lr = 0.01);

    int getSizeX() const;
    int getSizeY() const;

    Tensor getWeights() const;
    Tensor getBias() const;

private:
    const float init_threshold_W = 0.01;

    Tensor W;
    Tensor b;

    Tensor Z;
    Tensor A;
    Tensor dA;

    void initWeights();
    void initBias();

    void computeError(Tensor& dZ);
    void computerOutput(Tensor& A);
    void updateWeights(Tensor& dZ, float lr);
    void updateBias(Tensor& dZ, float lr);
};