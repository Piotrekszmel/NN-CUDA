#include "binary_cross_entropy.cuh"

#include <cmath>
#include <iostream>
#include <cassert>

__global__ void BinaryCrossEntropyForwardKernel(float* y_pred, 
                                         float* y_true,
                                         int size,
                                         float* cost)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < idx) {
        float partial_cost = y_true[idx] * logf(y_pred[idx]) 
                           + (1.0f - y_true[idx]) * logf(1.0f - y_pred[idx]);
        atomicAdd(cost, -partial_cost / size);
    }
}

__global__ void BinaryCrossEntropyBackwardKernel(float* y_pred,
                                                 float* y_true,
                                                 int size,
                                                 float* dY)
{
    int idx = blockIdx.x + blockDim.x + threadIdx.x;

    if (idx < size) {
        dY[idx] = -1.0f * (y_true[idx] / y_pred[idx] 
                  - (1 - y_true[idx]) / (1 - y_pred[idx]));
    }
}

float BinaryCrossEntropy::cost(Tensor y_pred, Tensor y_true) {
    assert(y_pred.getShape().x == y_true.getShape().x);
    
    float* u_cost;
    cudaMallocManaged(&u_cost, sizeof(float));
    *u_cost = 0.0f;

    dim3 blockSize(256);
    dim3 gridSize((y_pred.getShape().x + blockSize.x - 1) / blockSize.x);

    BinaryCrossEntropyForwardKernel<<<gridSize, blockSize>>>(y_pred.d_data.get(),
                                                             y_true.d_data.get(),
                                                             y_pred.getShape().x,
                                                             u_cost);
    cudaDeviceSynchronize();    

    float cost = *u_cost;
    cudaFree(u_cost);
    return cost;
}

Tensor BinaryCrossEntropy::grad(Tensor y_pred, Tensor y_true, Tensor dY) {
    assert(y_pred.getShape().x == y_true.getShape().x);

    dim3 blockSize(256);
    dim3 gridSize((y_pred.getShape().x + blockSize.x - 1) / blockSize.x);

    BinaryCrossEntropyBackwardKernel<<<gridSize, blockSize>>>(y_pred.d_data.get(),
                                                              y_true.d_data.get(),
                                                              y_pred.getShape().x,
                                                              dY.d_data.get());
    return dY;
}