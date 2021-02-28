#include <iostream>
#include <cstdlib>
#include <cassert>
#include <random>

#include "linear.cuh"
#include "../utils/utils.cuh"

__global__ void linearForwardKernel(float* W,
                                    float* A,
                                    float* b,
                                    int w_size_x,
                                    int w_size_y,
                                    int a_size_x,
                                    int a_size_y,
                                    float* Z)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    if (row < w_size_y && col < a_size_x) {
        for (int i = 0; i < w_size_x; i++) {
            sum += W[row * w_size_x + i] * A[i * a_size_x + col];
        }
        Z[row * a_size_x + col] = sum + b[row];
    }
}

__global__ void linearBackwardKernel(float* W,
                                     float* gradients,
                                     int w_size_x,
                                     int w_size_y,
                                     int g_size_x,
                                     int g_size_y,
                                     float* grad)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0;

    if (row < w_size_x && col < g_size_x) {
        for (int i = 0; i < w_size_y; i++) {
            sum += W[i * w_size_x + row] * gradients[i * g_size_x + col];
        }
        grad[row * g_size_x + col] = sum;
    }
}

__global__ void updateWeightsKernel(float* gradients, 
                                    float* A,
                                    int g_size_x,
                                    int g_size_y,
                                    int a_size_x,
                                    int a_size_y,
                                    float lr,
                                    float* W)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    if (row < g_size_y && col < a_size_y) {
        for (int i = 0; i < g_size_x; i++) {
            sum += gradients[row * g_size_x + i] * A[col * a_size_x + i];
        }
        W[row * a_size_y + col] = W[row * a_size_y + col] - lr * (sum / a_size_x);
    }
}

__global__ void updateBiasKernel(float* gradients, 
                                 int g_size_x,
                                 int g_size_y,
                                 int b_size_x,
                                 float lr,
                                 float* b)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < g_size_x * g_size_y) {
        int grad_x = idx % g_size_x;
        int grad_y = idx / g_size_x;
        atomicAdd(&b[grad_y], 
                  - lr * (gradients[grad_y * g_size_x + grad_x] / g_size_x));
    }
}
                                 

Linear::Linear(std::string name, Shape W_shape)
    : W(W_shape), b(W_shape.y, 1)
{
    this->name = name;
    W.allocMem();
    b.allocMem();
    initWeights();
    initBias();
}

void Linear::initWeights() {
    std::default_random_engine gen;
    std::normal_distribution<float> normal_dist(0.0, 1.0);

    for (int i = 0; i < W.getShape().y; i++) {
        for (int j = 0; j < W.getShape().x; j++) {
            W[i * W.getShape().x + j] = normal_dist(gen) * init_threshold_W;
        }
    }

    W.memCpy(HostToDevice);
}

void Linear::initBias() {
    for (int i = 0; i < b.getShape().x; i++) {
        b[i] = 0.0f;
    }

    b.memCpy(HostToDevice);
}

Tensor& Linear::forward(Tensor& A) {
    assert(W.getShape().x == A.getShape().y);

    this->A = A;
    Shape Z_shape(A.getShape().x, W.getShape().y);

    Z.allocMem(Z_shape);
    computeOutput(A);

    return Z;
}

void Linear::computeOutput(Tensor& A) {
    dim3 blockSize(8, 8);
    dim3 gridSize((Z.getShape().x + blockSize.x - 1) / blockSize.x,
                   (Z.getShape().y + blockSize.y - 1) / blockSize.y);
    linearForwardKernel<<<gridSize, blockSize>>>(W.d_data.get(),
                                                 A.d_data.get(),
                                                 b.d_data.get(),
                                                 W.getShape().x,
                                                 W.getShape().y,
                                                 A.getShape().x,
                                                 A.getShape().y,
                                                 Z.d_data.get());
}


Tensor& Linear::backward(Tensor& gradients, float lr) {
    dA.allocMem(A.getShape());

    computeError(gradients);
    
    updateBias(gradients, lr);
    updateWeights(gradients, lr);

    return dA;
}

void Linear::computeError(Tensor& gradients) {
    dim3 blockSize(8, 8);
    dim3 gridSize((A.getShape().x + blockSize.x - 1) / blockSize.x,
                  (A.getShape().y + blockSize.y - 1) / blockSize.y);
    linearBackwardKernel<<<gridSize, blockSize>>>(W.d_data.get(),
                                            gradients.d_data.get(),
                                            W.getShape().x,
                                            W.getShape().y,
                                            gradients.getShape().x,
                                            gradients.getShape().y,
                                            dA.d_data.get());
}

void Linear::updateWeights(Tensor& gradients, float lr) {
    dim3 blockSize(8, 8);
    dim3 gridSize((W.getShape().x + blockSize.x - 1) / blockSize.x,
                  (W.getShape().y + blockSize.y - 1) / blockSize.y);
    updateWeightsKernel<<<gridSize, blockSize>>>(gradients.d_data.get(),
                                                 A.d_data.get(),
                                                 gradients.getShape().x,
                                                 gradients.getShape().y,
                                                 A.getShape().x, 
                                                 A.getShape().y,
                                                 lr,
                                                 W.d_data.get());
}

void Linear::updateBias(Tensor& gradients, float lr) {
    dim3 blockSize(256);
    dim3 gridSize((gradients.getShape().x * gradients.getShape().y + blockSize.x - 1) / blockSize.x);
    updateBiasKernel<<<gridSize, blockSize>>>(gradients.d_data.get(),
                                              gradients.getShape().x,
                                              gradients.getShape().y,
                                              b.getShape().x,
                                              lr,
                                              b.d_data.get());
}

int Linear::getSizeX() const {
    return W.getShape().x;
}

int Linear::getSizeY() const {
    return W.getShape().y;
}

Tensor Linear::getWeights() const {
    return W;
}

Tensor Linear::getBias() const {
    return b;
}

