#include <iostream>
#include <cstdlib>
#include <cassert>
#include <random>

#include "linear.cuh"
#include "../utils/utils.cuh"

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

    for (int i = 0; i < W.shape.y; i++) {
        for (int j = 0; j < W.shape.x; j++) {
            W[i * W.shape.x + j] = normal_dist(gen) * init_threshold_W;
        }
    }

    W.memCpy(HostToDevice);
}

void Linear::initBias() {
    for (int i = 0; i < W.shape.x; i++) {
        b[i] = 0.0f;
    }

    b.memCpy(HostToDevice);
}
