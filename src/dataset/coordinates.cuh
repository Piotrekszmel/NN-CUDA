#pragma once 

#include <fstream>
#include <vector>
#include <string>

#include "../Tensor/tensor.cuh"

class Coordinates {
public:
    Coordinates(size_t batch_size, size_t num_batches);
    
    int getNumBatches();
    std::vector<Tensor>& getBatches();
    std::vector<Tensor>& getTargets();
    void saveToFile(Tensor& batch,
                    Tensor& labels,
                    std::string path0 = "src/dataset/coordinates_output_zero.txt",
                    std::string path1 = "src/dataset/coordinates_output_one.txt");
private:
    size_t batch_size;
    size_t num_batches;

    std::vector<Tensor> batches;
    std::vector<Tensor> targets;
};

