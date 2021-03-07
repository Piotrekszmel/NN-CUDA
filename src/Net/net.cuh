#pragma once

#include <vector>

#include "../layers/layer.hpp"
#include "../loss/binary_cross_entropy.cuh"
#include "../dataset/coordinates.cuh"

class Net {
public:
    Net(float lr = 0.01);
    ~Net();
    
    Tensor forward(Tensor X);
    void backward(Tensor y_pred, Tensor y_true);
    void train(size_t batch_size, size_t num_batches, size_t num_epochs);
    void addLayer(Layer* layer);
    
    float accuracy(const Tensor& predictions, const Tensor& targets);
    std::vector<Layer*> getLayers() const;

private:
    std::vector<Layer*> layers;
    BinaryCrossEntropy cost_fn;

    Tensor Y;
    Tensor dY;
    float lr;
};