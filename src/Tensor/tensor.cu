#include "tensor.cuh"

Tensor::Tensor(size_t x_, size_t y_)
    : shape(x_, y_), d_data(nullptr),
      h_data(nullptr), d_alloc(false),
      h_alloc(false) {}

Tensor::Tensor(Shape shape) 
    : Tensor(shape.x, shape.y) {}

void Tensor::allocDevice() {
    if (!d_alloc) {
        float* d_mem = nullptr;
        gpuErrCheck(cudaMalloc(&d_mem, shape.x * shape.y * sizeof(float)));

        d_data = std::shared_ptr<float>(d_mem, [&](float* ptr){ cudaFree(ptr); });
        d_alloc = true;
    }
}

void Tensor::allocHost() {
    if (!h_alloc) {
        h_data = std::shared_ptr<float>(new float[shape.x * shape.y], 
                                        [&](float* ptr){ delete[] ptr; });
    }
    h_alloc = true;
}

void Tensor::allocMem() {
    allocDevice();
    allocHost();
}

void Tensor::allocMem(Shape shape) {
    if (!d_alloc && !h_alloc) {
        this->shape = shape;
        allocMem();
    } else {
        printf("Memory already allocated!\n");
    }
}

void Tensor::memCpy(CopyType ct) {
    if (ct == HostToDevice) {
        if (d_alloc && h_alloc) {
            cudaMemcpy(d_data.get(), h_data.get(), shape.x * shape.y * sizeof(float),
                       cudaMemcpyHostToDevice);
        } else {
            perror("Cannot copy data from host to device!");
            exit(1);
        }
    } else if (ct == DeviceToHost) {
        if (d_alloc && h_alloc) {
            cudaMemcpy(h_data.get(), d_data.get(), shape.x * shape.y * sizeof(float),
                       cudaMemcpyDeviceToHost);

        } else {
            perror("Cannot copy data from device to host!");
            exit(1);
        }
    } else {
        perror("Wrong copy type!");
        exit(1);
    }
}

float& Tensor::operator[](const int idx) {
    return h_data.get()[idx];
}

const float& Tensor::operator[](const int idx) const {
    return h_data.get()[idx];
}

Shape Tensor::getShape() {
    return this->shape;
}