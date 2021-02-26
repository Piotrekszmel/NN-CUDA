#include "net.cuh"

Net::Net(float lr) : lr(lr) {}

Net::~Net() {
    for (auto layer : this->layers) {
        delete layer;
    }
}

void Net::addLayer(Layer* layer) {
    this->layers.push_back(layer);
}

Tensor Net::forward(Tensor X) {
    Tensor Z = X;

    for (auto layer : this->layers) {
        Z = layer->forward(Z);
    }

    Y = Z;
    return Y;
}

void Net::backward(Tensor y_pred, Tensor y_true) {
    dY.allocMem();

    Tensor error = cost_fn.grad(y_pred, y_true, dY);

    for (auto iter = this->layers.rbegin(); iter != this->layers.rend(); iter++) {
        error = (*iter)->backward(error, lr);
    }

    cudaDeviceSynchronize();
}

std::vector<Layer*> Net::getLayers() const {
    return layers;
}