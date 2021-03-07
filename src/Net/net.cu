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
    dY.allocMem(y_pred.getShape());

    Tensor error = cost_fn.grad(y_pred, y_true, dY);
    
    for (auto iter = this->layers.rbegin(); iter != this->layers.rend(); iter++) {
        error = (*iter)->backward(error, lr);
    }

    cudaDeviceSynchronize();
}

std::vector<Layer*> Net::getLayers() const {
    return layers;
}

void Net::train(size_t batch_size, size_t num_batches, size_t num_epochs) {
    Coordinates dataset(batch_size, num_batches);
    BinaryCrossEntropy cost_fn;

    Tensor Y;
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float cost = 0.0;

        for (int batch = 0; batch < dataset.getNumBatches() - 1; batch++) {
            Y = forward(dataset.getBatches().at(batch));
            backward(Y, dataset.getTargets().at(batch));
            cost += cost_fn.cost(Y, dataset.getTargets().at(batch));
        }
        if (epoch % 100 == 0) {
            std::cout << "Epochs(" << epoch << ") COST: " 
            << cost / dataset.getNumBatches()
            << std::endl;
        }
    } 

    Y = forward(dataset.getBatches().at(dataset.getNumBatches() - 1));
    Y.memCpy(DeviceToHost);
    
    dataset.saveToFile(dataset.getBatches().at(dataset.getNumBatches() - 1), Y);

    float acc = accuracy(Y, dataset.getTargets().at(dataset.getNumBatches() - 1));
    
    std::cout << "Accuracy: " << acc << std::endl;
}

float Net::accuracy(const Tensor& predictions, const Tensor& targets) {
    int m = predictions.getShape().x;
	int correct_predictions = 0;

	for (int i = 0; i < m; i++) {
		float prediction = predictions[i] > 0.5 ? 1 : 0;
		if (prediction == targets[i]) {
			correct_predictions++;
		}
	}
    
	return static_cast<float>(correct_predictions) / m;
}