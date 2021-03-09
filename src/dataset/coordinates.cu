#include "coordinates.cuh"

Coordinates::Coordinates(size_t batch_size, size_t num_batches) :
	batch_size(batch_size), num_batches(num_batches)
{
	std::ofstream targets_zero_f("src/dataset/coordinates_targets_one.txt");
	std::ofstream targets_one_f("src/dataset/coordinates_targets_zero.txt");

	for (int i = 0; i < num_batches; i++) {
		batches.push_back(Tensor(Shape(batch_size, 2)));
		targets.push_back(Tensor(Shape(batch_size, 1)));

		batches[i].allocMem();
		targets[i].allocMem();

		for (int k = 0; k < batch_size; k++) {
			int offset = batches[i].getShape().x;
			
			batches[i][k] = static_cast<float>(rand()) / RAND_MAX - 0.5;
			batches[i][k + offset] = static_cast<float>(rand()) / RAND_MAX - 0.5;;
			if ( (batches[i][k] > 0 && batches[i][k + offset] > 0) ||
				 ((batches[i][k] < 0 && batches[i][k + offset] < 0)) ) {
				targets[i][k] = 1;
				targets_one_f << batches[i][k] << " " << batches[i][k + offset] << "\n";
			}
			else {
				targets[i][k] = 0;
				targets_zero_f << batches[i][k] << " " << batches[i][k + offset] << "\n";
			}
		}

		batches[i].memCpy(HostToDevice);
		targets[i].memCpy(HostToDevice);
	}
	targets_zero_f.close();
	targets_one_f.close();
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

void Coordinates::saveToFile(Tensor& batch,
							 Tensor& labels,
							 std::string path0,
							 std::string path1) {
	std::ofstream targets_zero_f(path0);
	std::ofstream targets_one_f(path1);

	for (int k = 0; k < batch_size; k++) {
		int offset = batch.getShape().x;

		if (labels[k] >= 0.5) {
			targets_one_f << batch[k] << " " << batch[k + offset] << "\n";
		}
		else {
			targets_zero_f << batch[k] << " " << batch[k + offset] << "\n";
		}
	}
}