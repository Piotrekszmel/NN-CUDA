#pragma once

#include "../Tensor./tensor.cuh"

class BinaryCrossEntropy {
public:
    float cost(Tensor y_pred, Tensor y_true);
    Tensor grad(Tensor y_pred, Tensor y_true, Tensor dY);
};