#include <iostream>
#include <ctime>

#include "Net/net.cuh"
#include "layers/linear.cuh"
#include "activations/relu.cuh"
#include "activations/sigmoid.cuh"
#include "loss/binary_cross_entropy.cuh"

#include "Tensor/tensor.cuh"
#include "dataset/coordinates.cuh"

float computeAccuracy(const Tensor& y_pred, const Tensor& y_true);

int main()
{
    srand(time(NULL));

    Coordinates dataset(100, 21);
    BinaryCrossEntropy cost_fn;
    
    Net net;
    net.addLayer(new Linear("linear_1", Shape(2, 30)));
    net.addLayer(new ReLU("relu_1"));
    net.addLayer(new Linear("linear_2", Shape(30, 1)));
    net.addLayer(new Sigmoid("sigmoid_output"));

    Tensor Y;
    for (int epoch = 0; epoch < 1000; epoch++) {
        float cost = 0.0;

        for (int batch = 0; batch < dataset.getNumBatches() - 1; batch++) {
            Y = net.forward(dataset.getBatches().at(batch));
            net.backward(Y, dataset.getTargets().at(batch));
            cost += cost_fn.cost(Y, dataset.getTargets().at(batch));
        }
        if (epoch % 100 == 0) {
            std::cout << "Epochs(" << epoch << ") COST: " 
            << cost / dataset.getNumBatches()
            << std::endl;
        }
    } 

    Y = net.forward(dataset.getBatches().at(dataset.getNumBatches() - 1));
    Y.memCpy(DeviceToHost);
    
    dataset.saveToFile(dataset.getBatches().at(dataset.getNumBatches() - 1), Y);

    float accuracy = computeAccuracy(Y, dataset.getTargets().at(dataset.getNumBatches() - 1));
    
    std::cout << "Accuracy: " << accuracy << std::endl;
    
    return 0;
}

float computeAccuracy(const Tensor& predictions, const Tensor& targets) {
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