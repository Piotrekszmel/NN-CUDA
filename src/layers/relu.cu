#include "relu.cuh"

__global__ void reluForwardKernel(float* Z,
                                  int z_size_x,
                                  int z_size_y,
                                  float* A)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < z_size_x * z_size_y) {
        A[idx] = fmaxf(Z[idx], 0.0f);
    }
}

__global__ void reluBackwardKernel(float* Z,
                                   float* dA,
                                   int z_size_x,
                                   int z_size_y,
                                   float* dZ)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < z_size_x * z_size_y) {
        if (Z[idx] > 0) {
            dZ[idx] = dA[idx];
        } else {
            dZ[idx] = 0.0f;
        }
    }
}

ReLU::ReLU(std::string name) {
    this->name = name;
}

Tensor& ReLU::forward(Tensor& Z) {
    this->Z = Z;

    A.allocMem(Z.getShape());

    dim3 blockSize(256);
    dim3 gridSize((Z.getShape().x * Z.getShape().y + blockSize.x - 1) / blockSize.x);

    reluForwardKernel<<<gridSize, blockSize>>>(Z.d_data.get(),
                                               Z.getShape().x,
                                               Z.getShape().y,
                                               A.d_data.get());
    return A;
}

Tensor& ReLU::backward(Tensor& dA, float lr) {
    dZ.allocMem(Z.getShape());

    dim3 blockSize(256);
    dim3 gridSize((Z.getShape().x * Z.getShape().y + blockSize.x - 1) / blockSize.x);

    reluBackwardKernel<<<gridSize, blockSize>>>(Z.d_data.get(),
                                                dA.d_data.get(),
                                                Z.getShape().x,
                                                Z.getShape().y,
                                                dZ.d_data.get());
    return dZ;
}
