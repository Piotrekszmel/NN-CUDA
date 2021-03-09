#include "sigmoid.cuh"

__device__ float sigmoid(float x) {
    return 1.0f / (1 + exp(-x));
}

__global__ void sigmoidForwardKernel(float* Z, 
                                     int z_size_x,
                                     int z_size_y,
                                     float* A) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < z_size_x * z_size_y) {
        A[idx] = sigmoid(Z[idx]);
    }
}

__global__ void sigmoidBackwardKernel(float* Z, 
                                      float* dA,
                                      int z_size_x,
                                      int z_size_y,
                                      float* dZ)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < z_size_x * z_size_y) {
        dZ[idx] = dA[idx] * sigmoid(Z[idx]) * (1 - sigmoid(Z[idx]));
    }
}

Tensor& Sigmoid::forward(Tensor& Z) {
    this->Z = Z;
    A.allocMem(Z.getShape());

    dim3 blockSize(256);
    dim3 gridSize((Z.getShape().x * Z.getShape().y + blockSize.x - 1) / blockSize.x);

    sigmoidForwardKernel<<<gridSize, blockSize>>>(Z.d_data.get(),
                                                  Z.getShape().x,
                                                  Z.getShape().y,
                                                  A.d_data.get());
    return A;
}

Tensor& Sigmoid::backward(Tensor& dA, float lr) {
    dZ.allocMem(Z.getShape());

    dim3 blockSize(256);
    dim3 gridSize((Z.getShape().x * Z.getShape().y + blockSize.x - 1) / blockSize.x);

    sigmoidBackwardKernel<<<gridSize, blockSize>>>(Z.d_data.get(),
                                                   dA.d_data.get(),
                                                   Z.getShape().x,
                                                   Z.getShape().y,
                                                   dZ.d_data.get());
    return dZ;
}

void Sigmoid::info() {
    std::cout << "Sigmoid: A -> " << A.getShape() << "  Z -> " << Z.getShape() << "\n"; 
}