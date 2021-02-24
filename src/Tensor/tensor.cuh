#pragma once

#include <memory>

#include "../utils/shape.cuh"
#include "../utils/utils.cuh"

enum CopyType 
{
    HostToDevice,
    DeviceToHost
};

class Tensor
{
public:
    Tensor(size_t x_ = 1, size_t y_ = 1);
    Tensor(Shape shape);

    void allocMem();
    void allocMem(Shape shape);

    void memCpy(CopyType ct = HostToDevice);

    Shape getShape() const;
    
    float& operator[](const int idx);
    const float& operator[](const int idx) const;

    std::shared_ptr<float> d_data;
    std::shared_ptr<float> h_data;

private:
    Shape shape;
    bool d_alloc;
    bool h_alloc;

    void allocDevice();
    void allocHost();
};