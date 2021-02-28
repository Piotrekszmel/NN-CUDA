#include "coordinates.cuh"

Coordinates::Coordinates(size_t batch_size, size_t num_batches) :
	batch_size(batch_size), num_batches(num_batches)
{
	for (int i = 0; i < num_batches; i++) {
		batches.push_back(Tensor(Shape(batch_size, 2)));
		targets.push_back(Tensor(Shape(batch_size, 1)));

		batches[i].allocMem();
		targets[i].allocMem();

		for (int k = 0; k < batch_size; k++) {
			batches[i][k] = static_cast<float>(rand()) / RAND_MAX - 0.5;
			batches[i][batches[i].getShape().x + k] = static_cast<float>(rand()) / RAND_MAX - 0.5;;

			if ( (batches[i][k] > 0 && batches[i][batches[i].getShape().x + k] > 0) ||
				 ((batches[i][k] < 0 && batches[i][batches[i].getShape().x + k] < 0)) ) {
				targets[i][k] = 1;
			}
			else {
				targets[i][k] = 0;
			}
		}

		batches[i].memCpy(HostToDevice);
		targets[i].memCpy(HostToDevice);
	}
}

int Coordinates::getNumBatches() {
	return num_batches;
}

std::vector<Tensor>& Coordinates::getBatches() {
	return batches;
}

std::vector<Tensor>& Coordinates::getTargets() {
	return targets;
}