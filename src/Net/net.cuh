#pragma once

#include <vector>

#include "../layers/layer.hpp"
#include "../loss/binary_cross_entropy.cuh"

class Net {
public:
    Net(float lr = 0.01);
    ~Net();
    Tensor forward(Tensor X);
    void backward(Tensor y_pred, Tensor y_true);

    void addLayer(Layer* layer);
    std::vector<Layer*> getLayers() const;

private:
    std::vector<Layer*> layers;
    BinaryCrossEntropy cost_fn;

    Tensor Y;
    Tensor dY;
    float lr;
};