#pragma once 

#include <vector>

#include "../Tensor/tensor.cuh"

class Coordinates {
public:
    Coordinates(size_t batch_size, size_t num_batches);
    
    int getNumBatches();
    std::vector<Tensor>& getBatches();
    std::vector<Tensor>& getTargets();

private:
    size_t batch_size;
    size_t num_batches;

    std::vector<Tensor> batches;
    std::vector<Tensor> targets;
};

